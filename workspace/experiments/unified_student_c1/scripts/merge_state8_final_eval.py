#!/usr/bin/env python3
"""Merge four disjoint state8 formal shards and three frozen C4 pair seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_formal_protocols import summarize
from merge_state8_promotion_eval import merge_c4
from t3_judge_utils import read_jsonl, write_jsonl


FROZEN_TOTAL = 21676
FROZEN_T3 = 1494


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def expected_ids(eval_jsonl: Path) -> set[str]:
    values = {str(row["example_id"]) for row in read_jsonl(eval_jsonl)}
    if len(values) != FROZEN_TOTAL:
        raise ValueError(f"Frozen formal support drift: {len(values)}")
    return values


def merge_formal(
    predictions: list[Path], metrics: list[Path], eval_jsonl: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(predictions) != len(metrics) or len(metrics) < 2:
        raise ValueError("Expected matching multi-shard predictions and metrics")
    shard_count = len(metrics)
    metric_rows = [read(path) for path in metrics]
    coordinates = {
        (int(row["evaluation_shard"]["index"]), int(row["evaluation_shard"]["count"]))
        for row in metric_rows
    }
    if coordinates != {(index, shard_count) for index in range(shard_count)}:
        raise ValueError(f"Formal shard coordinate drift: {sorted(coordinates)}")
    baseline = metric_rows[0]
    invariants = (
        "model",
        "adapter",
        "recurrent_state",
        "state_architecture",
        "state_shape",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "unsharded_filtered_rows",
        "artifact_sha256",
    )
    for row in metric_rows[1:]:
        for key in invariants:
            if row.get(key) != baseline.get(key):
                raise ValueError(f"Formal shard invariant drift: {key}")
    if baseline.get("state_architecture") != "layerwise":
        raise ValueError("Final evaluator is not layerwise")
    if baseline.get("state_shape") != [64, 8, 5120]:
        raise ValueError(f"Final state shape drift: {baseline.get('state_shape')}")
    if baseline.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Final decoder can see raw history")
    if baseline.get("persistent_history_kv_cache") is not False:
        raise ValueError("Final decoder retains history KV cache")
    if int(baseline.get("unsharded_filtered_rows", -1)) != FROZEN_TOTAL:
        raise ValueError("Final evaluator denominator drift")
    if sum(int(row.get("sampled_rows", -1)) for row in metric_rows) != FROZEN_TOTAL:
        raise ValueError("Formal shard support does not sum to 21,676")

    records = [item for path in predictions for item in read_jsonl(path)]
    ids = [str(item["example_id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs across final formal shards")
    expected = expected_ids(eval_jsonl)
    observed = set(ids)
    if observed != expected:
        raise ValueError(
            f"Final formal coverage drift missing={len(expected-observed)} "
            f"extra={len(observed-expected)}"
        )
    records.sort(key=lambda row: str(row["example_id"]))
    t3 = [row for row in records if row.get("task") == "diagnose_misconception"]
    if len(t3) != FROZEN_T3:
        raise ValueError(f"Frozen T3 support drift: {len(t3)}")
    result = {
        "stage": "state8_final_formal_merge_v1",
        "sampled_rows": len(records),
        "state_architecture": "layerwise",
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "artifact_sha256": baseline["artifact_sha256"],
        "metrics_by_group": summarize(records),
        "shards": [
            {
                "coordinate": row["evaluation_shard"],
                "sampled_rows": int(row["sampled_rows"]),
            }
            for row in sorted(
                metric_rows, key=lambda value: int(value["evaluation_shard"]["index"])
            )
        ],
    }
    return records, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--c4", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, formal = merge_formal(args.predictions, args.metrics, args.eval_jsonl)
    artifacts = formal["artifact_sha256"]
    c4 = merge_c4(args.c4, artifacts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    write_jsonl(
        args.output_dir / "t3_predictions.jsonl",
        [row for row in records if row.get("task") == "diagnose_misconception"],
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(formal, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "c4_multiseed.json").write_text(
        json.dumps(c4, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"formal": formal, "c4": c4}, sort_keys=True))


if __name__ == "__main__":
    main()
