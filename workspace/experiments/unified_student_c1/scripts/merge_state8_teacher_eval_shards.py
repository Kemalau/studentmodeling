#!/usr/bin/env python3
"""Fail-closed merge for N qualified-teacher T3 generation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_formal_protocols import read_jsonl, summarize, write_jsonl


EXPECTED_SUPPORT = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 894,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 300,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 300,
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def expected_ids(eval_jsonl: Path) -> set[str]:
    rows = read_jsonl(eval_jsonl)
    ids = {
        str(row["example_id"])
        for row in rows
        if row.get("task") == "diagnose_misconception"
        and str((row.get("metadata") or {}).get("eval_source"))
        in {"eedi_mining", "malrule"}
    }
    if len(ids) != 1494:
        raise ValueError(f"Frozen teacher support drift: {len(ids)}")
    return ids


def merge(
    prediction_paths: list[Path], metric_paths: list[Path], eval_jsonl: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard_count = len(prediction_paths)
    if shard_count < 2 or len(metric_paths) != shard_count:
        raise ValueError(
            "Expected equal prediction/metric shard counts with at least two shards"
        )
    metrics = [load_json(path) for path in metric_paths]
    coordinates = {
        (int(row["evaluation_shard"]["index"]), int(row["evaluation_shard"]["count"]))
        for row in metrics
    }
    if coordinates != {(index, shard_count) for index in range(shard_count)}:
        raise ValueError(f"Teacher shard-coordinate drift: {sorted(coordinates)}")
    invariants = (
        "model",
        "adapter",
        "eval_jsonl",
        "artifact_sha256",
        "max_samples_per_group",
        "max_length",
        "max_new_tokens",
        "generation_batch_size",
        "bottleneck_mask",
        "bottleneck_style",
        "unsharded_filtered_rows",
    )
    baseline = metrics[0]
    for row in metrics[1:]:
        for key in invariants:
            if row.get(key) != baseline.get(key):
                raise ValueError(f"Teacher shard invariant drift for {key}")
    if baseline.get("unsharded_filtered_rows") != 1494:
        raise ValueError("Teacher shard did not filter the frozen 1494-row panel")
    if baseline.get("max_samples_per_group") is not None:
        raise ValueError("Teacher shard unexpectedly sampled the formal panel")
    records = [row for path in prediction_paths for row in read_jsonl(path)]
    ids = [str(row["example_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate teacher prediction IDs across shards")
    expected = expected_ids(eval_jsonl)
    observed = set(ids)
    if expected != observed:
        raise ValueError(
            "Teacher shard coverage mismatch "
            f"missing={sorted(expected - observed)[:3]} "
            f"extra={sorted(observed - expected)[:3]}"
        )
    records.sort(key=lambda row: str(row["example_id"]))
    summary = summarize(records)
    observed_support = {
        key: int(value["n"])
        for key, value in summary.items()
    }
    if observed_support != EXPECTED_SUPPORT:
        raise ValueError(f"Teacher group support drift: {observed_support}")
    merged = {
        "stage": "state8_full_history_teacher_sharded_merge_v1",
        "sampled_rows": len(records),
        "generation_rows": len(records),
        "evaluation_shards": shard_count,
        "artifact_sha256": baseline["artifact_sha256"],
        "decoding": {
            "max_length": baseline["max_length"],
            "max_new_tokens": baseline["max_new_tokens"],
            "generation_batch_size": baseline["generation_batch_size"],
        },
        "metrics_by_group": summary,
    }
    return records, merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, metrics = merge(args.predictions, args.metrics, args.eval_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
