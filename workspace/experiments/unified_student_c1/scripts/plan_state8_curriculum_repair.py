#!/usr/bin/env python3
"""Derive bounded repairs when every five-stage State8 checkpoint fails.

The main curriculum matrix contains two information routes and four immutable
checkpoints per route.  This planner never weakens a gate or opens decoder
history.  It classifies the failed matrix, chooses the strongest checkpoint
for each independent failure axis, and emits at most three one-GPU state-only
continuations that can run in parallel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

try:
    from select_state8_curriculum_checkpoint import ARMS, STEPS, evaluate
except ImportError:  # package-style tests
    from scripts.select_state8_curriculum_checkpoint import ARMS, STEPS, evaluate


EXPECTED = {(arm, step) for arm in ARMS for step in STEPS}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def choose(
    candidates: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> dict[str, Any]:
    return max(candidates, key=key)


def source(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": "curriculum",
        "arm": row["curriculum_arm"],
        "checkpoint_step": row["checkpoint_step"],
    }


def plan(values: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [evaluate(value) for value in values]
    coordinates = {
        (row["curriculum_arm"], row["checkpoint_step"]) for row in candidates
    }
    if coordinates != EXPECTED:
        raise ValueError(f"Curriculum screen matrix drift: {sorted(coordinates)}")
    if any(row["qualified"] for row in candidates):
        raise ValueError("A qualified curriculum checkpoint must be promoted, not repaired")

    def minimum(row: dict[str, Any], names: tuple[str, ...]) -> float:
        return min(float(row["checks"][name]) for name in names)

    semantic = choose(
        candidates,
        lambda row: (
            minimum(
                row,
                (
                    "T3_semantic",
                    "T3_eedi_semantic",
                    "T3_synthetic_cross_semantic",
                    "T3_synthetic_same_semantic",
                    "T3_schema",
                ),
            ),
            row["score"],
            row["checkpoint_step"],
            row["curriculum_arm"] == "current_query",
        ),
    )
    specificity = choose(
        candidates,
        lambda row: (
            minimum(
                row,
                (
                    "answer_C4_D_specific",
                    "malrule_diagnosis_C4_D_specific",
                    "eedi_same_question_C4_D_specific",
                ),
            ),
            row["checks"]["eedi_same_question_C4_swap_to_donor"],
            -row["checks"]["eedi_same_question_C4_random_stable"],
            row["score"],
        ),
    )
    controls = choose(
        candidates,
        lambda row: (
            minimum(
                row,
                (
                    "malrule_diagnosis_C4_noop_retention",
                    "malrule_diagnosis_C4_same_rule_retention",
                    "eedi_same_question_C4_noop_retention",
                    "eedi_same_question_C4_same_rule_retention",
                ),
            ),
            -row["checks"]["eedi_same_question_C4_random_stable"],
            row["score"],
        ),
    )
    retention = choose(
        candidates,
        lambda row: (
            minimum(
                row,
                (
                    "T1_macro_f1",
                    "T2_math_equivalence",
                    "T5_joint_math_equivalence",
                    "T6_macro_f1",
                ),
            ),
            row["checks"]["T6_positive_recall"],
            row["score"],
        ),
    )

    failed = {name for row in candidates for name in row["failed_checks"]}
    symptoms = {
        "semantic_capacity": bool(
            failed
            & {
                "T3_semantic",
                "T3_eedi_semantic",
                "T3_synthetic_cross_semantic",
                "T3_synthetic_same_semantic",
                "T3_schema",
            }
        ),
        "task_retention": bool(
            failed
            & {
                "T1_macro_f1",
                "T2_math_equivalence",
                "T5_joint_math_equivalence",
                "T6_macro_f1",
                "T6_positive_recall",
            }
        ),
        "causal_specificity": bool(
            failed
            & {
                "answer_C4_pass",
                "malrule_diagnosis_C4_pass",
                "eedi_same_question_C4_pass",
            }
        ),
        "identity_controls": max(
            minimum(
                row,
                (
                    "malrule_diagnosis_C4_noop_retention",
                    "malrule_diagnosis_C4_same_rule_retention",
                    "eedi_same_question_C4_noop_retention",
                    "eedi_same_question_C4_same_rule_retention",
                ),
            )
            for row in candidates
        )
        < 0.60,
    }

    branches: list[dict[str, Any]] = []
    if symptoms["semantic_capacity"]:
        branches.append(
            {
                "name": "semantic_binding",
                "objective": "restore natural-language diagnosis under the state bottleneck",
                "source": source(semantic),
                "training_arm": "full",
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "200",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                    "STATE8_CAUSAL_WEIGHT": "1.5",
                    "STATE8_CAUSAL_MARGIN": "0.5",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
                },
            }
        )
    if symptoms["causal_specificity"]:
        branches.append(
            {
                "name": "specificity_triplet",
                "objective": "separate matched state swaps from random-state shortcuts",
                "source": source(specificity),
                # The random-state control is supervised with the explicit
                # insufficient-evidence decode target.  Updater-only training
                # can move the state geometry, but cannot teach that decoder
                # behavior.  Keep this branch full just like the proactive
                # checkpoint-4000 specificity intervention.
                "training_arm": "full",
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "150",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
                    "STATE8_CAUSAL_WEIGHT": "3.0",
                    "STATE8_CAUSAL_MARGIN": "0.75",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
                },
            }
        )
    # A full specificity branch already contains both random-neutral decoding
    # and same-rule state geometry, so it also exercises the identity
    # controls.  With a three-job cap, reserve the remaining independent slot
    # for task retention when that gate is failing rather than duplicating the
    # causal-control axis.
    if (
        symptoms["identity_controls"]
        and not symptoms["causal_specificity"]
        and len(branches) < 3
    ):
        branches.append(
            {
                "name": "identity_controls",
                "objective": "stabilize no-op and same-rule interventions",
                "source": source(controls),
                "training_arm": "updater",
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "200",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
                    "STATE8_CAUSAL_WEIGHT": "2.0",
                    "STATE8_CAUSAL_MARGIN": "0.5",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
                },
            }
        )
    if symptoms["task_retention"] and len(branches) < 3:
        branches.append(
            {
                "name": "balanced_continuation",
                "objective": "retain T1/T2/T5/T6 while consolidating state-only decoding",
                "source": source(retention),
                "training_arm": "full",
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "100",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                    "STATE8_CAUSAL_WEIGHT": "2.0",
                    "STATE8_CAUSAL_MARGIN": "0.5",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
                },
            }
        )
    branches = branches[:3]
    if not branches:
        # A failure can in principle come from a newly introduced hard gate.
        # Continue the best overall checkpoint with the conservative balanced arm.
        best = max(candidates, key=lambda row: (row["score"], row["checkpoint_step"]))
        branches.append(
            {
                "name": "balanced_continuation",
                "objective": "conservative state-only continuation for an unclassified hard-gate failure",
                "source": source(best),
                "training_arm": "full",
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "100",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                    "STATE8_CAUSAL_WEIGHT": "2.0",
                    "STATE8_CAUSAL_MARGIN": "0.5",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
                },
            }
        )

    return {
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "status": "complete",
        "diagnostic_only": True,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "state_token_count": 8,
        "symptoms": symptoms,
        "failed_checks": sorted(failed),
        "branches": branches,
        "parallel_gpu_jobs": len(branches),
        "candidate_audit": sorted(
            candidates,
            key=lambda row: (row["curriculum_arm"], row["checkpoint_step"]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan([read(path) for path in args.candidate])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
