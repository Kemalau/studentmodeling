#!/usr/bin/env python3
"""Merge Qwen candidate-scoring results into the baseline evaluation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_simple_t1_t6_baselines import TASKS, markdown_matrix, svg_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--qwen-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen3-32B mixed QLoRA")
    args = parser.parse_args()

    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    qwen = json.loads(args.qwen_json.read_text(encoding="utf-8"))
    qwen_metrics = qwen["model"]["metrics"]
    by_task_source = {}
    for task in TASKS:
        for source, metrics in qwen_metrics[task].get("by_source", {}).items():
            by_task_source[f"{task}/{source}"] = metrics
    baseline["models"][args.model_name] = {
        "overall": {},
        "by_task": {task: qwen_metrics[task] for task in TASKS},
        "by_task_source": by_task_source,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation_matrix_with_qwen.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "evaluation_matrix.md").write_text(markdown_matrix(baseline), encoding="utf-8")
    (args.out_dir / "evaluation_matrix.svg").write_text(svg_matrix(baseline), encoding="utf-8")
    print(markdown_matrix(baseline))


if __name__ == "__main__":
    main()
