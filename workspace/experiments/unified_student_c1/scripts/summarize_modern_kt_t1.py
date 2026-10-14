#!/usr/bin/env python3
"""Aggregate frozen five-source modern-KT results without hiding seed failures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for path in sorted(args.input_dir.rglob("metrics.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            continue
        grouped[value["architecture"]].append(value)
    if not grouped:
        raise ValueError(f"No completed results under {args.input_dir}")
    summary = {"stage": "modern_kt_t1_five_source_summary_v1", "status": "complete", "models": {}}
    lines = [
        "# Modern KT on frozen five-source T1",
        "",
        "Cells report mean ± sample standard deviation across seeds. Every seed-level value remains in the JSON artifact.",
        "",
        "| Model | Seeds | Overall Acc | Overall AUROC | Real-only Acc | Real-only AUROC | Temporal Acc | Temporal AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, values in sorted(grouped.items()):
        values.sort(key=lambda item: int(item["seed"]))
        record = {"seeds": [int(item["seed"]) for item in values], "seed_results": values, "aggregate": {}}
        for group in ("overall", "real_all", "real_temporal"):
            source_values = [item["test"] if group == "overall" else item["test"][group] for item in values]
            record["aggregate"][group] = {}
            for metric in ("accuracy", "auc"):
                array = np.asarray([float(item[metric]) for item in source_values])
                record["aggregate"][group][metric] = {
                    "mean": float(array.mean()),
                    "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                }
        summary["models"][model_name] = record

        def render(group: str, metric: str) -> str:
            item = record["aggregate"][group][metric]
            return f"{item['mean']:.3f} ± {item['sd']:.3f}"

        lines.append(
            f"| {model_name} | {len(values)} | {render('overall', 'accuracy')} | {render('overall', 'auc')} | "
            f"{render('real_all', 'accuracy')} | {render('real_all', 'auc')} | "
            f"{render('real_temporal', 'accuracy')} | {render('real_temporal', 'auc')} |"
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
