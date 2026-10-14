#!/usr/bin/env python3
"""Fail-closed merge for sharded state8 T3 and C4 promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_formal_protocols import summarize
from t3_judge_utils import read_jsonl, write_jsonl


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def expected_t3_ids(eval_jsonl: Path) -> set[str]:
    rows = read_jsonl(eval_jsonl)
    ids = {
        str(row["example_id"])
        for row in rows
        if row.get("task") == "diagnose_misconception"
        and str((row.get("metadata") or row.get("original_metadata") or {}).get("eval_source") or row.get("source"))
        in {"eedi_mining", "malrule"}
    }
    if len(ids) != 1494:
        raise ValueError(f"Frozen T3 support drift: expected 1494, observed {len(ids)}")
    return ids


def merge_t3(
    predictions: list[Path], metrics: list[Path], eval_jsonl: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(predictions) != len(metrics) or len(predictions) != 4:
        raise ValueError("Expected exactly four T3 prediction/metrics shards")
    records = [row for path in predictions for row in read_jsonl(path)]
    ids = [str(row["example_id"]) for row in records]
    duplicate_ids = sorted({example_id for example_id in ids if ids.count(example_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate T3 prediction IDs: {duplicate_ids[:3]}")
    expected = expected_t3_ids(eval_jsonl)
    observed = set(ids)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"T3 shard coverage mismatch missing={missing[:3]} extra={extra[:3]}"
        )
    metric_rows = [load_json(path) for path in metrics]
    shard_coordinates = {
        (int(value["evaluation_shard"]["index"]), int(value["evaluation_shard"]["count"]))
        for value in metric_rows
    }
    if shard_coordinates != {(index, 4) for index in range(4)}:
        raise ValueError(f"T3 shard-coordinate drift: {sorted(shard_coordinates)}")
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
    baseline = metric_rows[0]
    for value in metric_rows[1:]:
        for key in invariants:
            if value.get(key) != baseline.get(key):
                raise ValueError(f"T3 shard invariant drift for {key}")
    if baseline.get("state_shape") != [64, 8, 5120]:
        raise ValueError(f"Unexpected state8 shape: {baseline.get('state_shape')}")
    if baseline.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("T3 decoder history route is open")
    if baseline.get("persistent_history_kv_cache") is not False:
        raise ValueError("T3 decoder has persistent history KV cache")
    records.sort(key=lambda row: str(row["example_id"]))
    by_source: dict[str, int] = {}
    for row in records:
        source = str(row["source"])
        by_source[source] = by_source.get(source, 0) + 1
    if by_source != {"eedi_mining": 894, "malrule": 600}:
        raise ValueError(f"T3 source support drift: {by_source}")
    summary = {
        "stage": "state8_promotion_t3_merge_v1",
        "sampled_rows": len(records),
        "source_counts": by_source,
        "state_shape": baseline["state_shape"],
        "artifact_sha256": baseline["artifact_sha256"],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "metrics_by_group": summarize(records),
    }
    return records, summary


def merge_c4(
    paths: list[Path], expected_artifacts: dict[str, str] | None = None
) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError("Expected exactly three C4 results")
    rows = []
    for path in paths:
        value = load_json(path)
        artifacts = value.get("artifact_sha256") or {}
        if expected_artifacts is not None and artifacts != expected_artifacts:
            raise ValueError(f"C4 checkpoint hash drift in {path}: {artifacts}")
        summary = value["summary"]
        clean = summary["clean"]
        seed = int(value.get("pair_seed", value.get("seed", -1)))
        rows.append(
            {
                "pair_seed": seed,
                "clean_n": int(summary["clean_n"]),
                "swap": float(clean["swap"]["flip_to_A_rate"]),
                "random": float(clean["random"]["flip_to_A_rate"]),
                "same_rule": float(clean["samemal"]["flip_to_A_rate"]),
                "noop": float(clean["noop"]["flip_to_A_rate"]),
                "D_specific": float(summary["D_specific"]),
                "scientific_gate_pass": bool(summary["scientific_gate_pass"]),
            }
        )
        if value.get("state_shape") != [64, 8, 5120]:
            raise ValueError(f"Unexpected C4 state shape in {path}")
        if value.get("raw_history_visible_to_decoder") is not False:
            raise ValueError(f"C4 decoder history route is open in {path}")
        if value.get("persistent_history_kv_cache") is not False:
            raise ValueError(f"C4 decoder has persistent history KV cache in {path}")
    if {row["pair_seed"] for row in rows} != {13, 42, 97}:
        raise ValueError(f"C4 seed set drift: {rows}")
    rows.sort(key=lambda row: row["pair_seed"])
    return {
        "stage": "state8_promotion_c4_multiseed_v1",
        "rows": rows,
        "mean_D_specific": sum(row["D_specific"] for row in rows) / len(rows),
        "passing_seeds": sum(row["scientific_gate_pass"] for row in rows),
        "replication_gate_pass": (
            sum(row["scientific_gate_pass"] for row in rows) >= 2
            and sum(row["D_specific"] for row in rows) / len(rows) > 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs=4, required=True)
    parser.add_argument("--t3-metrics", type=Path, nargs=4, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--c4", type=Path, nargs=3, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--recurrent-state-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, t3_summary = merge_t3(
        args.predictions, args.t3_metrics, args.eval_jsonl
    )
    expected_artifacts = {
        "adapter_model": args.adapter_sha256,
        "recurrent_state": args.recurrent_state_sha256,
    }
    if t3_summary.get("artifact_sha256") != expected_artifacts:
        raise ValueError(
            f"T3 checkpoint hash drift: {t3_summary.get('artifact_sha256')}"
        )
    c4_summary = merge_c4(args.c4, expected_artifacts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "t3_predictions.jsonl", records)
    (args.output_dir / "t3_metrics.json").write_text(
        json.dumps(t3_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "c4_multiseed.json").write_text(
        json.dumps(c4_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"t3": t3_summary, "c4": c4_summary}, sort_keys=True))


if __name__ == "__main__":
    main()
