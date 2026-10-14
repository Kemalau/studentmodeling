#!/usr/bin/env python3
"""Turn a failed four-checkpoint State8 screen into bounded repair branches.

This planner is intentionally torch-free.  It does not promote a diagnostic
checkpoint and it does not weaken any scientific floor.  It identifies the
dominant failure modes, chooses an immutable checkpoint source, and emits at
most three orthogonal experiments for the Feeds repair controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from select_state8_eedi_same_question_checkpoint import evaluate
except ImportError:  # package-style tests
    from scripts.select_state8_eedi_same_question_checkpoint import evaluate


EXPECTED = {(arm, step) for arm in ("updater", "full") for step in (20, 100)}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _minimum(row: dict[str, Any], names: tuple[str, ...]) -> float:
    return min(float(row["checks"][name]) for name in names)


def _source(candidates: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    if objective == "semantic":
        key = lambda row: (
            row["checks"]["T3_eedi_semantic"],
            _minimum(
                row,
                ("T3_synthetic_cross_semantic", "T3_synthetic_same_semantic"),
            ),
            row["score"],
            -row["step"],
        )
    elif objective == "specificity":
        key = lambda row: (
            row["checks"]["eedi_D_specific"],
            row["checks"]["eedi_swap_to_donor"],
            -row["checks"]["eedi_random_stable"],
            row["score"],
            -row["step"],
        )
    elif objective == "controls":
        key = lambda row: (
            min(
                row["checks"]["eedi_noop_retention"],
                row["checks"]["eedi_same_rule_retention"],
            ),
            row["checks"]["eedi_native_to_recipient"],
            -row["checks"]["eedi_random_stable"],
            row["score"],
            -row["step"],
        )
    else:
        raise ValueError(f"Unknown repair objective: {objective}")
    return max(candidates, key=key)


def _late_erosion(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["arm"], row["step"]): row for row in candidates}
    metrics = (
        "T3_eedi_semantic",
        "eedi_D_specific",
        "eedi_swap_to_donor",
        "eedi_same_rule_retention",
        "eedi_noop_retention",
    )
    details: dict[str, Any] = {}
    detected = False
    for arm in ("updater", "full"):
        early, late = by_key[(arm, 20)], by_key[(arm, 100)]
        deltas = {
            name: float(late["checks"][name]) - float(early["checks"][name])
            for name in metrics
        }
        score_delta = float(late["score"]) - float(early["score"])
        # One large scientific regression or a broad score regression is
        # enough to avoid using the later checkpoint as a repair source.
        arm_detected = min(deltas.values()) <= -0.15 or score_delta <= -0.08
        detected = detected or arm_detected
        details[arm] = {
            "detected": arm_detected,
            "score_delta_100_minus_20": score_delta,
            "metric_deltas_100_minus_20": deltas,
        }
    return {"detected": detected, "by_arm": details}


def plan(values: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [evaluate(value) for value in values]
    keys = {(row["arm"], row["step"]) for row in candidates}
    if keys != EXPECTED:
        raise ValueError(f"Checkpoint matrix drift: {sorted(keys)}")
    if any(row["hard_sanity"] for row in candidates):
        raise ValueError("Repair planning received a viable early checkpoint")

    semantic_source = _source(candidates, "semantic")
    specificity_source = _source(candidates, "specificity")
    controls_source = _source(candidates, "controls")
    erosion = _late_erosion(candidates)
    best_eedi = max(row["checks"]["T3_eedi_semantic"] for row in candidates)
    best_synthetic = max(
        _minimum(
            row,
            ("T3_synthetic_cross_semantic", "T3_synthetic_same_semantic"),
        )
        for row in candidates
    )
    best_specificity = max(row["checks"]["eedi_D_specific"] for row in candidates)
    best_swap = max(row["checks"]["eedi_swap_to_donor"] for row in candidates)
    lowest_random = min(row["checks"]["eedi_random_stable"] for row in candidates)
    best_identity = max(
        min(
            row["checks"]["eedi_noop_retention"],
            row["checks"]["eedi_same_rule_retention"],
        )
        for row in candidates
    )
    symptoms = {
        "eedi_semantic_below_floor": best_eedi < 0.20,
        "synthetic_semantic_below_floor": best_synthetic < 0.40,
        "eedi_specificity_below_floor": best_specificity < 0.10,
        "swap_binding_below_floor": best_swap < 0.40,
        "random_control_leakage": lowest_random > 0.40,
        "identity_control_instability": best_identity < 0.60,
        "late_training_erosion": erosion["detected"],
    }

    branches: list[dict[str, Any]] = []
    if symptoms["eedi_semantic_below_floor"] or symptoms[
        "synthetic_semantic_below_floor"
    ]:
        branches.append(
            {
                "name": "semantic_binding",
                "objective": "raise natural-language semantic T3 without opening history",
                "source": {
                    "arm": semantic_source["arm"],
                    "checkpoint_step": semantic_source["step"],
                },
                "training_arm": "full",
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "200",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                    "STATE8_CAUSAL_WEIGHT": "1.5",
                    "STATE8_CAUSAL_MARGIN": "0.5",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
                },
                "rationale": (
                    "Increase same-question semantic coverage while reducing the "
                    "causal coefficient relative to the current repair; the decoder "
                    "still sees only state plus current item."
                ),
            }
        )
    if (
        symptoms["eedi_specificity_below_floor"]
        or symptoms["swap_binding_below_floor"]
        or symptoms["random_control_leakage"]
    ):
        branches.append(
            {
                "name": "specificity_triplet",
                "objective": "separate donor swap from norm-matched random state",
                "source": {
                    "arm": specificity_source["arm"],
                    "checkpoint_step": specificity_source["step"],
                },
                "training_arm": "updater",
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "150",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
                    "STATE8_CAUSAL_WEIGHT": "3.0",
                    "STATE8_CAUSAL_MARGIN": "0.75",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
                },
                "rationale": (
                    "Train only the state updater with a stronger swap-minus-random "
                    "triplet; freezing the answer path limits semantic drift."
                ),
            }
        )
    if symptoms["identity_control_instability"]:
        branches.append(
            {
                "name": "identity_controls",
                "objective": "stabilize same-rule and no-op recipient predictions",
                "source": {
                    "arm": controls_source["arm"],
                    "checkpoint_step": controls_source["step"],
                },
                "training_arm": "updater",
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "200",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
                    "STATE8_CAUSAL_WEIGHT": "2.0",
                    "STATE8_CAUSAL_MARGIN": "0.5",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
                },
                "rationale": (
                    "Replay more recipient/same-rule twins at ordinary margin; this "
                    "targets invariance instead of merely increasing swap pressure."
                ),
            }
        )

    # Keep the sweep bounded even when every metric fails.  Semantic capacity,
    # donor specificity, and identity invariance are the three independent axes.
    branches = branches[:3]
    if not branches:
        raise RuntimeError("No repair branch was derived from a failed matrix")
    return {
        "stage": "state8_eedi_same_question_failed_checkpoint_repair_plan_v1",
        "status": "complete",
        "diagnostic_only": True,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "state_token_count": 8,
        "symptoms": symptoms,
        "late_training_erosion": erosion,
        "branches": branches,
        "candidate_audit": sorted(candidates, key=lambda row: (row["arm"], row["step"])),
        "parallel_gpu_jobs": len(branches),
    }


def plan_extension(selection: dict[str, Any]) -> dict[str, Any]:
    """Continue one viable early checkpoint through the full repair protocol.

    A 20/100-step screen is diagnostic only: it does not contain the complete
    T1/T2/T5/T6 evidence required for promotion.  Treating it as "no repair
    required" used to leave the pipeline with no way to promote that early
    checkpoint if both long-running parent arms later eroded.  This bounded
    continuation creates exactly one independently evaluated candidate.
    """
    if selection.get("stage") != "state8_eedi_same_question_early_checkpoint_selection_v1":
        raise ValueError("Early-selection stage drift")
    selected = selection.get("selected_for_extended_training") or {}
    arm, step = str(selected.get("arm") or ""), int(selected.get("step", -1))
    if arm not in {"updater", "full"} or step not in {20, 100}:
        raise ValueError("No viable early checkpoint to extend")
    if selected.get("hard_sanity") is not True:
        raise ValueError("Selected early checkpoint did not pass hard sanity")
    return {
        "stage": "state8_eedi_same_question_early_checkpoint_extension_plan_v1",
        "status": "complete",
        "diagnostic_only": True,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "state_token_count": 8,
        "branches": [
            {
                "name": "balanced_continuation",
                "objective": "complete unified and dual-C4 evaluation from the viable early checkpoint",
                "source": {"arm": arm, "checkpoint_step": step},
                "training_arm": arm,
                "overrides": {
                    "STATE8_EEDI_CAUSAL_TWINS": "100",
                    "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                    "STATE8_CAUSAL_WEIGHT": "2.0",
                    "STATE8_CAUSAL_MARGIN": "0.5",
                    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
                },
                "rationale": (
                    "Preserve the passed early semantic/causal balance while "
                    "obtaining the full promotion evidence under the unchanged "
                    "strict state-only protocol."
                ),
            }
        ],
        "parallel_gpu_jobs": 1,
        "source_selection": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=Path)
    parser.add_argument("--early-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.candidate) == bool(args.early_selection):
        raise ValueError("Specify either --candidate or --early-selection")
    result = (
        plan([read(path) for path in args.candidate])
        if args.candidate
        else plan_extension(read(args.early_selection))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
