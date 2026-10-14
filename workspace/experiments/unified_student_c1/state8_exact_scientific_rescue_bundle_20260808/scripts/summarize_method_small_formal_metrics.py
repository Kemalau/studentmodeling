#!/usr/bin/env python3
"""Reduce method-small formal protocol groups to the T1/T2/T3/T5/T6 headline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_METRICS = {
    "predict_correctness": ("T1", "accuracy"),
    "predict_response": ("T2", "math_equivalence"),
    "diagnose_misconception": ("T3", "normalized_exact_match"),
    "predict_prospective_error": ("T5", "joint_math_equivalence"),
    "predict_error_persistence": ("T6", "accuracy"),
}


def summarize(payload: dict) -> dict:
    groups = payload["metrics_by_group"]
    output = {}
    for task, (short_name, metric_name) in TASK_METRICS.items():
        selected = [
            (name, values)
            for name, values in groups.items()
            if name.split("/", 1)[0] == task and metric_name in values
        ]
        support = sum(int(values["n"]) for _, values in selected)
        weighted = sum(
            int(values["n"]) * float(values[metric_name])
            for _, values in selected
        ) / max(1, support)
        output[short_name] = {
            "task": task,
            "metric": metric_name,
            "value": weighted,
            "n": support,
            "n_protocol_groups": len(selected),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
