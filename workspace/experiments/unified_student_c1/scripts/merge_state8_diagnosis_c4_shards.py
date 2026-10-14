#!/usr/bin/env python3
"""Merge disjoint diagnosis-C4 shards and recompute corpus-level exact metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_recurrent_diagnosis_c4 import CONDITIONS, summarize_exact
from t3_judge_utils import write_jsonl


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge(shards: list[Path], expected_pairs: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    judge_predictions: list[dict[str, Any]] = []
    judge_eval: list[dict[str, Any]] = []
    metrics = [read_json(path / "metrics.json") for path in shards]
    if len({json.dumps(row["artifact_sha256"], sort_keys=True) for row in metrics}) != 1:
        raise ValueError("Shard artifact hash drift")
    for path in shards:
        records.extend(read_jsonl(path / "predictions.jsonl"))
        judge_predictions.extend(read_jsonl(path / "judge_predictions.jsonl"))
        judge_eval.extend(read_jsonl(path / "judge_eval.jsonl"))
    pair_ids = sorted({int(row["pair_idx"]) for row in records})
    if pair_ids != list(range(1, expected_pairs + 1)):
        raise ValueError(f"Pair coverage drift: {pair_ids}")
    record_keys = [(int(row["pair_idx"]), str(row["condition"])) for row in records]
    expected_record_keys = [
        (pair_idx, condition)
        for pair_idx in range(1, expected_pairs + 1)
        for condition in CONDITIONS
    ]
    if sorted(record_keys) != sorted(expected_record_keys):
        raise ValueError("Missing or duplicate pair-condition records")
    prediction_ids = [str(row["example_id"]) for row in judge_predictions]
    eval_ids = [str(row["example_id"]) for row in judge_eval]
    if len(set(prediction_ids)) != expected_pairs * len(CONDITIONS) * 2:
        raise ValueError("Judge prediction ID coverage drift")
    if sorted(prediction_ids) != sorted(eval_ids):
        raise ValueError("Judge prediction/eval alignment drift")
    records.sort(key=lambda row: (int(row["pair_idx"]), CONDITIONS.index(row["condition"])))
    judge_predictions.sort(key=lambda row: str(row["example_id"]))
    judge_eval.sort(key=lambda row: str(row["example_id"]))
    return {
        "records": records,
        "judge_predictions": judge_predictions,
        "judge_eval": judge_eval,
        "metrics": {
            "stage": "state8_closed_diagnosis_c4_merged_v1",
            "num_pairs": expected_pairs,
            "conditions": list(CONDITIONS),
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "artifact_sha256": metrics[0]["artifact_sha256"],
            "exact_label_auxiliary": summarize_exact(records),
            "semantic_judgment_pending": True,
            "shards": len(shards),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.shard, args.expected_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", result["records"])
    write_jsonl(
        args.output_dir / "judge_predictions.jsonl", result["judge_predictions"]
    )
    write_jsonl(args.output_dir / "judge_eval.jsonl", result["judge_eval"])
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
