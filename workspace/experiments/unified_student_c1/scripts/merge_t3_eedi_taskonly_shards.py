#!/usr/bin/env python3
"""Fail-closed merge for the four frozen Eedi-only T3 specialist shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_formal_protocols import read_jsonl, summarize, write_jsonl


EXPECTED_ADAPTER = "329e95ad02a10cd9541f81ce3f220aaa521369f5761659a146fc2c54ad859bb2"
EXPECTED_EVAL = "bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def expected_ids(eval_jsonl: Path) -> set[str]:
    ids = {
        str(row["example_id"])
        for row in read_jsonl(eval_jsonl)
        if row.get("task") == "diagnose_misconception"
        and str((row.get("metadata") or {}).get("eval_source")) == "eedi_mining"
    }
    if len(ids) != 894:
        raise ValueError(f"Frozen Eedi support drift: {len(ids)}")
    return ids


def merge(
    prediction_paths: list[Path], metric_paths: list[Path], eval_jsonl: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(prediction_paths) != 4 or len(metric_paths) != 4:
        raise ValueError("Expected exactly four prediction and metric shards")
    metrics = [load_json(path) for path in metric_paths]
    coordinates = {
        (int(row["evaluation_shard"]["index"]), int(row["evaluation_shard"]["count"]))
        for row in metrics
    }
    if coordinates != {(index, 4) for index in range(4)}:
        raise ValueError(f"Shard-coordinate drift: {sorted(coordinates)}")
    invariants = (
        "model", "adapter", "eval_jsonl", "artifact_sha256",
        "max_samples_per_group", "max_length", "max_new_tokens",
        "generation_batch_size", "bottleneck_mask", "bottleneck_style",
        "unsharded_filtered_rows",
    )
    baseline = metrics[0]
    for row in metrics[1:]:
        for key in invariants:
            if row.get(key) != baseline.get(key):
                raise ValueError(f"Shard invariant drift for {key}")
    if baseline.get("unsharded_filtered_rows") != 894:
        raise ValueError("Shard did not filter the frozen 894-row Eedi panel")
    if baseline.get("max_samples_per_group") is not None:
        raise ValueError("Formal shard unexpectedly sampled the panel")
    hashes = baseline.get("artifact_sha256") or {}
    if hashes.get("adapter_model") != EXPECTED_ADAPTER or hashes.get("evaluation_input") != EXPECTED_EVAL:
        raise ValueError("Frozen artifact hash drift")
    records = [row for path in prediction_paths for row in read_jsonl(path)]
    ids = [str(row["example_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction IDs across shards")
    expected = expected_ids(eval_jsonl)
    observed = set(ids)
    if observed != expected:
        raise ValueError(
            f"Coverage mismatch missing={sorted(expected-observed)[:3]} extra={sorted(observed-expected)[:3]}"
        )
    if any(row.get("task") != "diagnose_misconception" or row.get("source") != "eedi_mining" for row in records):
        raise ValueError("Non-Eedi/T3 row in specialist merge")
    records.sort(key=lambda row: str(row["example_id"]))
    grouped = summarize(records)
    key = "diagnose_misconception/eedi_mining/open_taxonomy_generation"
    if set(grouped) != {key} or int(grouped[key]["n"]) != 894:
        raise ValueError(f"Group support drift: {grouped}")
    summary = {
        "stage": "t3_eedi_taskonly_qwen_sharded_merge_v1",
        "status": "complete",
        "sampled_rows": 894,
        "generation_rows": 894,
        "evaluation_shards": 4,
        "artifact_sha256": hashes,
        "decoding": {
            "max_length": baseline["max_length"],
            "max_new_tokens": baseline["max_new_tokens"],
            "generation_batch_size": baseline["generation_batch_size"],
        },
        "metrics_by_group": grouped,
    }
    return records, summary


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
