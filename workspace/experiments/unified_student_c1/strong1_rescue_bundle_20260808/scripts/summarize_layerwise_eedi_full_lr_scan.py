#!/usr/bin/env python3
"""Compare a matched pair of task-retention learning-rate arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_LRS = {2e-6, 5e-6}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def summarize(
    paths: list[Path], *, loss_profile: str = "state_path", branch: str = "eedi_full"
) -> dict[str, Any]:
    if loss_profile not in {"state_path", "state_kd"}:
        raise ValueError("Unsupported full-Eedi loss profile")
    rows = [read(path) for path in paths]
    if len(rows) != 2 or {float(row.get("learning_rate", -1)) for row in rows} != EXPECTED_LRS:
        raise ValueError("Eedi full LR arm-set drift")
    if any(
        row.get("branch") != branch
        or row.get("loss_profile") != loss_profile
        for row in rows
    ):
        raise ValueError("Eedi full recipe drift")
    eligible = [row for row in rows if row.get("all_gates_pass") is True]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["T3_semantic"]["retention_ratio"]),
            float(row["T6"]["retention_ratio"]),
            float(row["C4"]["D_specific"]),
            -float(row["learning_rate"]),
        ),
    ) if eligible else None
    return {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "diagnostic_only": True,
        "source_checkpoint_round": 750,
        "matched_data": True,
        "loss_profile": loss_profile,
        "arms": sorted(rows, key=lambda row: float(row["learning_rate"])),
        "any_all_gates_pass": bool(eligible),
        "selected_candidate": (
            {
                "candidate_id": selected["candidate_id"],
                "branch": selected["branch"],
                "loss_profile": selected["loss_profile"],
                "learning_rate": selected["learning_rate"],
            }
            if selected
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs=2, type=Path, required=True)
    parser.add_argument(
        "--loss-profile", choices=("state_path", "state_kd"), default="state_path"
    )
    parser.add_argument(
        "--branch",
        choices=("eedi_full", "eedi_full_open_t3", "equal_t3_t6_open_t3"),
        default="eedi_full",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.summaries, loss_profile=args.loss_profile, branch=args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
