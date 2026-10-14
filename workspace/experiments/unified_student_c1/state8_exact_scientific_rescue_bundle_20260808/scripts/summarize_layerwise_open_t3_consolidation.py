#!/usr/bin/env python3
"""Compare phase-2 consolidation arms and select only a full-gate pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def margin(row: dict) -> float:
    if row.get("pruned") is True:
        return 0.0
    return min(
        float(row["C4"]["D_specific"]) / 0.15,
        float(row["T3_semantic"]["retention_ratio"]) / 0.85,
        float(row["T6"]["retention_ratio"]) / 0.85,
        float(row["formal_screen"]["T3_format_valid_rate"]) / 0.90,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-kd", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [read(args.state_kd), read(args.state_path)]
    passing = [row for row in rows if row.get("all_gates_pass") is True]
    selected = max(passing, key=margin) if passing else None
    value = {
        "stage": "layerwise_open_t3_consolidation_selection_v1",
        "selection_rule": (
            "all frozen gates pass; maximize minimum normalized margin over "
            "C4 specificity, T3 semantic retention, T6 retention, T3 format"
        ),
        "all_gates_pass": selected is not None,
        "selected_candidate_id": (
            selected["candidate_id"] if selected is not None else None
        ),
        "selected_loss_profile": (
            selected["loss_profile"] if selected is not None else None
        ),
        "selected_learning_rate": (
            selected["learning_rate"] if selected is not None else None
        ),
        "rows": [
            ({
                "candidate_id": row["candidate_id"],
                "loss_profile": row["loss_profile"],
                "all_gates_pass": row["all_gates_pass"],
                "checks": row["checks"],
                "C4_D_specific": row["C4"]["D_specific"],
                "T3_format_valid_rate": row["formal_screen"][
                    "T3_format_valid_rate"
                ],
                "T3_semantic_retention": row["T3_semantic"]["retention_ratio"],
                "T6_retention": row["T6"]["retention_ratio"],
                "minimum_normalized_margin": margin(row),
            } if row.get("pruned") is not True else {
                "candidate_id": row["candidate_id"],
                "loss_profile": row["loss_profile"],
                "all_gates_pass": False,
                "checks": row["checks"],
                "pruned": True,
                "prune_reason": row["prune_reason"],
                "C4_D_specific": row["C4"]["D_specific"],
                "minimum_normalized_margin": 0.0,
            })
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
