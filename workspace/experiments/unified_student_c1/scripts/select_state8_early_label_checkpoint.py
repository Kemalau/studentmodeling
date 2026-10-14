#!/usr/bin/env python3
"""Select one early-label checkpoint at a single adaptive milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from select_state8_causal_teacher_checkpoint import (
        STEPS,
        evaluate as evaluate_checkpoint,
    )
    from select_state8_global_candidate import REPAIR_STAGE
except ImportError:  # package-style unit tests
    from scripts.select_state8_causal_teacher_checkpoint import (
        STEPS,
        evaluate as evaluate_checkpoint,
    )
    from scripts.select_state8_global_candidate import REPAIR_STAGE


SPECS = {
    (8301, "early_label_low"),
    (8302, "early_label_high"),
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def select(values: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [evaluate_checkpoint(value, SPECS) for value in values]
    steps = {int(row["checkpoint_step"]) for row in candidates}
    if len(steps) != 1 or not steps <= STEPS:
        raise ValueError(f"Early-label adaptive step drift: {sorted(steps)}")
    step = next(iter(steps))
    observed = {
        (row["repair_seed"], row["repair_branch"]) for row in candidates
    }
    if observed != SPECS:
        raise ValueError(f"Early-label checkpoint pair drift: {sorted(observed)}")
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(eligible, key=lambda row: (row["score"], row["arm"])) \
        if eligible else None
    return {
        "stage": REPAIR_STAGE,
        "status": "complete",
        "pass": selected is not None,
        "selected": selected,
        "selected_arm": selected["arm"] if selected else None,
        "candidates": sorted(candidates, key=lambda row: row["arm"]),
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "raw_history_visible_to_decoder": False,
        "state_token_count": 8,
        "answer_C4_confirmation_pending": selected is not None,
        "full_exact_epoch_pending": selected is not None,
        "adaptive_checkpoint_step": step,
        "selection_rule": (
            "first_20_100_240_500_milestone_with_any_candidate_passing_all_"
            "shared_task_semantic_and_dual_c4_gates"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select([read(path) for path in args.candidate])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
