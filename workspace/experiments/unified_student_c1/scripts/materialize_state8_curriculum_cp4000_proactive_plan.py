#!/usr/bin/env python3
"""Build three orthogonal State8 continuations at the state-only boundary.

Checkpoint 4000 is the first immutable checkpoint at the beginning of the
final history-dropout=1.0 quintile.  Starting bounded continuations there lets
the baseline curriculum finish unchanged while semantic binding, causal
specificity, and task retention are explored in parallel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARMS = {
    "generic": "state8_diagnosis_first_eedi_curriculum_20260812",
    "current_query": (
        "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
    ),
}


def read_receipt(
    results_root: Path, arm: str, checkpoint_step: int = 4000
) -> dict[str, Any]:
    if checkpoint_step not in {3000, 4000}:
        raise ValueError("Proactive checkpoint step must be 3000 or 4000")
    path = (
        results_root
        / ARMS[arm]
        / "milestones"
        / f"checkpoint-{checkpoint_step}.completion.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or int(value.get("step", -1)) != checkpoint_step
    ):
        raise ValueError(
            f"Incomplete checkpoint-{checkpoint_step} receipt: {path}"
        )
    hashes = value.get("sha256") or {}
    if set(hashes) != {"adapter_model", "recurrent_state"}:
        raise ValueError(f"Checkpoint hash schema drift: {path}")
    if any(
        not isinstance(item, str)
        or len(item) != 64
        or any(char not in "0123456789abcdef" for char in item)
        for item in hashes.values()
    ):
        raise ValueError(f"Invalid checkpoint hashes: {path}")
    return {"path": str(path), "sha256": hashes}


def build(
    results_root: Path,
    *,
    checkpoint_step: int = 4000,
    seed_base: int = 5201,
) -> dict[str, Any]:
    if checkpoint_step not in {3000, 4000}:
        raise ValueError("Proactive checkpoint step must be 3000 or 4000")
    if seed_base < 5201 or seed_base + 2 > 9999:
        raise ValueError("Proactive seed namespace is outside the audited range")
    receipts = {
        arm: read_receipt(results_root, arm, checkpoint_step) for arm in ARMS
    }
    branches = [
        {
            "name": "semantic_binding",
            "objective": (
                "bind natural-language diagnosis and evidence to query-aligned "
                "State8 without decoder history"
            ),
            "source": {
                "family": "curriculum",
                "arm": "current_query",
                "checkpoint_step": checkpoint_step,
            },
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                "STATE8_CAUSAL_WEIGHT": "1.5",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
            },
        },
        {
            "name": "specificity_triplet",
            "objective": (
                "maximize donor-state transfer while forcing random states to "
                "the insufficient-evidence target"
            ),
            "source": {
                "family": "curriculum",
                "arm": "current_query",
                "checkpoint_step": checkpoint_step,
            },
            # Full mode is intentional: unlike updater-only mode, it can train
            # the random-state insufficient-evidence decoder branch.
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "150",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
                "STATE8_CAUSAL_WEIGHT": "3.0",
                "STATE8_CAUSAL_MARGIN": "0.75",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
            },
        },
        {
            "name": "balanced_continuation",
            "objective": (
                "retain T1/T2/T5/T6 while consolidating strict state-only "
                "semantic and causal decoding"
            ),
            "source": {
                "family": "curriculum",
                "arm": "generic",
                "checkpoint_step": checkpoint_step,
            },
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "100",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                "STATE8_CAUSAL_WEIGHT": "2.0",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
            },
        },
    ]
    return {
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "status": "complete",
        "diagnostic_only": True,
        "proactive_checkpoint_step": checkpoint_step,
        "proactive_from_state_only_boundary": checkpoint_step == 4000,
        "proactive_before_state_only_boundary": checkpoint_step == 3000,
        "scientific_floors_unchanged": True,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "raw_history_visible_to_decoder": False,
        "state_token_count": 8,
        "random_state_target": "insufficient_evidence",
        "same_rule_geometry_required": True,
        # Keep proactive artifacts disjoint from the post-matrix repair
        # planner, whose curriculum namespace intentionally starts at 4201.
        "seed_base": seed_base,
        "source_receipts": receipts,
        "branches": branches,
        "parallel_gpu_jobs": len(branches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-step", type=int, choices=(3000, 4000), default=4000
    )
    parser.add_argument("--seed-base", type=int, default=5201)
    args = parser.parse_args()
    result = build(
        args.results_root,
        checkpoint_step=args.checkpoint_step,
        seed_base=args.seed_base,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
