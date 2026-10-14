#!/usr/bin/env python3
"""Select a T6-repair arm only after every frozen scientific gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    ("balanced", 5e-7): (200, 600),
    ("balanced", 1e-6): (200, 600),
    ("strong", 5e-7): (100, 700),
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def identify(row: dict[str, Any]) -> tuple[str, float]:
    counts = row.get("task_counts") or {}
    pair = (
        int(counts.get("diagnose_misconception", -1)),
        int(counts.get("predict_error_persistence", -1)),
    )
    variant = {value: key for key, value in {"balanced": (200, 600), "strong": (100, 700)}.items()}.get(pair)
    if variant is None:
        raise ValueError(f"Unknown T6-repair curriculum counts: {pair}")
    return variant, float(row.get("learning_rate", -1.0))


def normalized_margin(row: dict[str, Any]) -> float:
    return min(
        float(row["C4"]["D_specific"]) / 0.15,
        float(row["T3_semantic"]["retention_ratio"]) / 0.85,
        float(row["T6"]["retention_ratio"]) / 0.85,
        float(row["formal_screen"]["T3_format_valid_rate"]) / 0.90,
    )


def summarize(paths: list[Path]) -> dict[str, Any]:
    if not 1 <= len(paths) <= len(EXPECTED):
        raise ValueError("Expected one to three T6-repair summaries")
    arms: list[dict[str, Any]] = []
    observed: set[tuple[str, float]] = set()
    for path in paths:
        row = read(path)
        if row.get("branch") != "eedi_full_open_t3" or row.get("loss_profile") != "state_path":
            raise ValueError("T6-repair recipe drift")
        variant, learning_rate = identify(row)
        key = (variant, learning_rate)
        if key not in EXPECTED or key in observed:
            raise ValueError(f"T6-repair arm-set drift: {key}")
        observed.add(key)
        expected_t3, expected_t6 = EXPECTED[key]
        counts = row["task_counts"]
        if (
            int(counts["diagnose_misconception"]) != expected_t3
            or int(counts["predict_error_persistence"]) != expected_t6
        ):
            raise ValueError(f"T6-repair task-count drift: {key}")
        candidate_id = (
            f"eedi_full_open_t3__state_path__{variant}__lr{learning_rate:.0e}"
        )
        arms.append(
            {
                **row,
                "candidate_id": candidate_id,
                "curriculum_variant": variant,
                "minimum_normalized_margin": normalized_margin(row),
                "summary_path": str(path),
            }
        )
    missing = sorted(
        set(EXPECTED) - observed,
        key=lambda item: (item[0], item[1]),
    )
    selection_complete = not missing
    eligible = [row for row in arms if row.get("all_gates_pass") is True]
    selected = (
        max(
            eligible,
            key=lambda row: (
                float(row["minimum_normalized_margin"]),
                float(row["T6"]["retention_ratio"]),
                float(row["T3_semantic"]["retention_ratio"]),
                float(row["C4"]["D_specific"]),
                -float(row["learning_rate"]),
            ),
        )
        if eligible and selection_complete
        else None
    )
    ranked = sorted(
        arms,
        key=lambda row: (
            row.get("all_gates_pass") is True,
            float(row["minimum_normalized_margin"]),
        ),
        reverse=True,
    )
    return {
        "stage": "layerwise_eedi_open_t3_t6_repair_selection_v1",
        "diagnostic_only": True,
        "source_candidate": "eedi_full_open_t3__state_path__lr2e-06",
        "source_checkpoint_round": 4250,
        "final_checkpoint_round": 4500,
        "selection_rule": (
            "all frozen gates pass; maximize the minimum normalized margin over "
            "C4 specificity, T3 semantic retention, T6 retention, and T3 format"
        ),
        "selection_complete": selection_complete,
        "observed_arm_count": len(observed),
        "expected_arm_count": len(EXPECTED),
        "missing_arms": [
            {"curriculum_variant": variant, "learning_rate": learning_rate}
            for variant, learning_rate in missing
        ],
        "any_all_gates_pass": bool(eligible),
        "selected_candidate": (
            {
                "candidate_id": selected["candidate_id"],
                "curriculum_variant": selected["curriculum_variant"],
                "learning_rate": selected["learning_rate"],
                "summary_path": selected["summary_path"],
                "minimum_normalized_margin": selected["minimum_normalized_margin"],
            }
            if selected is not None
            else None
        ),
        "arms": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
