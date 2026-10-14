#!/usr/bin/env python3
"""Summarize the fixed-support round-1000 five-task diagnostic screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SUPPORT = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 25,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 25,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 25,
    "predict_correctness/dbe_kt22/temporal_warm_item_binary": 25,
    "predict_correctness/eedi_response/profile_heldout_binary": 25,
    "predict_correctness/foundationalassist/temporal_warm_item_binary": 25,
    "predict_correctness/malrule/synthetic_cross_template_binary": 25,
    "predict_correctness/malrule/synthetic_same_template_binary": 25,
    "predict_correctness/xes3g5m/temporal_warm_item_binary": 25,
    "predict_error_persistence/dbe_kt22/strict_exact_recurrence_binary": 12,
    "predict_error_persistence/foundationalassist/strict_exact_recurrence_binary": 25,
    "predict_prospective_error/dbe_kt22/delta3_closed_choice_generation/delta=3": 25,
    "predict_prospective_error/foundationalassist/delta_open_generation/delta=1": 25,
    "predict_prospective_error/foundationalassist/delta_open_generation/delta=3": 25,
    "predict_prospective_error/foundationalassist/delta_open_generation/delta=5": 25,
    "predict_response/dbe_kt22/closed_choice_generation": 25,
    "predict_response/eedi_response/closed_choice_generation": 25,
    "predict_response/foundationalassist/structured_math_open_generation": 25,
    "predict_response/malrule/synthetic_cross_template_open_generation": 25,
    "predict_response/malrule/synthetic_same_template_open_generation": 25,
}
EXPECTED_ROWS = sum(EXPECTED_SUPPORT.values())
THRESHOLDS = {
    "predict_correctness": ("macro_f1", 0.40),
    "predict_error_persistence": ("macro_f1", 0.25),
    "predict_response": ("math_equivalence", 0.20),
    "predict_prospective_error": ("joint_math_equivalence", 0.40),
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def weighted(groups: dict[str, Any], task: str, metric: str) -> float:
    rows = [value for name, value in groups.items() if name.split("/", 1)[0] == task]
    if not rows or any(metric not in row or int(row.get("n", 0)) <= 0 for row in rows):
        raise ValueError(f"Missing {metric} for {task}")
    support = sum(int(row["n"]) for row in rows)
    return sum(float(row[metric]) * int(row["n"]) for row in rows) / support


def summarize(root: Path, *, checkpoint_round: int = 1000) -> dict[str, Any]:
    if checkpoint_round <= 0:
        raise ValueError("checkpoint_round must be positive")
    paths = sorted(root.rglob("metrics.json"))
    # Feeds keeps T1/T6, T2, T3 and T5 in four independently recoverable jobs.
    # Local checkpoint diagnostics may combine these into two model loads, and
    # bounded rehearsal jobs use one model load. Exact protocol groups, support
    # and non-overlap below remain the authoritative completeness gate.
    if len(paths) not in {1, 2, 4}:
        raise ValueError(f"Expected one, two or four metric files: {paths}")
    groups: dict[str, Any] = {}
    sampled_rows = 0
    for path in paths:
        value = read(path)
        if value.get("state_architecture") != "layerwise" or tuple(value.get("state_shape") or ()) not in {
            (64, 5120), (64, 8, 5120)
        }:
            raise ValueError(f"Architecture drift: {path}")
        if value.get("raw_history_visible_to_decoder") is not False or value.get("persistent_history_kv_cache") is not False:
            raise ValueError(f"History leak: {path}")
        overlap = set(groups).intersection(value["metrics_by_group"])
        if overlap:
            raise ValueError(f"Duplicate groups: {overlap}")
        groups.update(value["metrics_by_group"])
        sampled_rows += int(value["sampled_rows"])
    exact_support = set(groups) == set(EXPECTED_SUPPORT) and all(
        int(groups[name].get("n", -1)) == support
        for name, support in EXPECTED_SUPPORT.items()
    )
    performance = {}
    for task, (metric, threshold) in THRESHOLDS.items():
        score = weighted(groups, task, metric)
        performance[task] = {
            "metric": metric,
            "score": score,
            "catastrophic_floor": threshold,
            "pass": score >= threshold,
        }
    t3_groups = [value for name, value in groups.items() if name.startswith("diagnose_misconception/")]
    t3_format = sum(float(row["format_valid_rate"]) * int(row["n"]) for row in t3_groups) / sum(
        int(row["n"]) for row in t3_groups
    )
    checks = {
        "exact_protocol_support": exact_support,
        "exact_sampled_rows": sampled_rows == EXPECTED_ROWS,
        "no_catastrophic_task_regression": all(row["pass"] for row in performance.values()),
        "T3_format_valid": t3_format >= 0.90,
    }
    return {
        "stage": f"layerwise_finalfit_checkpoint{checkpoint_round}_formal_screen",
        "checkpoint_round": checkpoint_round,
        "diagnostic_only": True,
        "formal_final_evaluation_required": True,
        "sampled_rows": sampled_rows,
        "performance": performance,
        "T3_format_valid_rate": t3_format,
        "checks": checks,
        "diagnostic_gate_pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-round", type=int, default=1000)
    args = parser.parse_args()
    value = summarize(args.root, checkpoint_round=args.checkpoint_round)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["diagnostic_gate_pass"] else 2)


if __name__ == "__main__":
    main()
