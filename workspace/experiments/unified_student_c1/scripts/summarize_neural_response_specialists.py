#!/usr/bin/env python3
"""Summarize formal neural response specialists without overclaiming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2", type=Path, required=True)
    parser.add_argument("--t4", type=Path, required=True)
    parser.add_argument("--traditional", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    t2 = load(args.t2)
    t4 = load(args.t4)
    traditional = load(args.traditional)
    matrix = load(args.matrix)
    rows = {row["model"]: row for row in matrix["rows"]}
    unified = rows["Unified Qwen3-32B QLoRA"]
    base = rows["Qwen3-32B Base"]
    values = {
        "T2": {
            "metric": "math_equivalence",
            "n": 3693,
            "neural_specialist": float(t2["metrics"]["overall"]["math_equivalence"]),
            "existing_specialist": float(traditional["T2"]["metrics"]["overall"]["math_equivalence"]),
            "base_qwen": float(base["T2"]["value"]),
            "unified_qwen": float(unified["T2"]["value"]),
        },
        "T4": {
            "metric": "joint_math_equivalence",
            "n": 4600,
            "neural_specialist": float(t4["metrics"]["overall"]["joint_math_equivalence"]),
            "existing_specialist": float(traditional["T5"]["metrics"]["overall"]["joint_math_equivalence"]),
            "base_qwen": float(base["T4"]["value"]),
            "unified_qwen": float(unified["T4"]["value"]),
        },
    }
    for cell in values.values():
        cell["neural_minus_existing"] = cell["neural_specialist"] - cell["existing_specialist"]
        cell["unified_minus_neural"] = cell["unified_qwen"] - cell["neural_specialist"]
        cell["interpretation"] = (
            "descriptively_stronger_than_existing_specialist"
            if cell["neural_minus_existing"] > 0
            else "does_not_improve_existing_specialist"
        )
        cell["significance_pending"] = True
    output = {
        "stage": "formal_neural_response_specialist_comparison_v1",
        "status": "complete",
        "cells": values,
        "claim_guard": (
            "Descriptive full-test comparison only. Do not call the neural specialist SOTA and do not claim a "
            "significant Unified advantage until paired per-example statistics are computed."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Formal neural response specialist comparison",
        "",
        "| Task | Metric | n | Neural specialist | Existing specialist | Base Qwen | Unified Qwen | Unified − neural |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, cell in values.items():
        lines.append(
            f"| {task} | {cell['metric']} | {cell['n']} | {cell['neural_specialist']:.3f} | "
            f"{cell['existing_specialist']:.3f} | {cell['base_qwen']:.3f} | "
            f"{cell['unified_qwen']:.3f} | {cell['unified_minus_neural']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "These are descriptive full-test comparisons. The response-aware GRU is a new neural specialist baseline, "
            "not a claimed public SOTA. Paired per-example uncertainty remains required before a significance claim.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
