from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from merge_state8_teacher_eval_shards import merge  # noqa: E402


GROUPS = (
    ("eedi_mining", "open_taxonomy_generation", 894),
    ("malrule", "synthetic_cross_template_open_generation", 300),
    ("malrule", "synthetic_same_template_open_generation", 300),
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def materialize(tmp_path: Path, shard_count: int = 4):
    rows = []
    for source, protocol, count in GROUPS:
        for index in range(count):
            rows.append(
                {
                    "example_id": f"{source}-{protocol}-{index:04d}",
                    "task": "diagnose_misconception",
                    "metadata": {"eval_source": source, "eval_protocol": protocol},
                    "target": "<label>x</label><diagnosis>x</diagnosis>",
                }
            )
    eval_jsonl = tmp_path / "eval.jsonl"
    write_jsonl(eval_jsonl, rows)
    prediction_paths = []
    metric_paths = []
    ordered = sorted(rows, key=lambda row: row["example_id"])
    for shard in range(shard_count):
        shard_rows = [
            row
            for position, row in enumerate(ordered)
            if position % shard_count == shard
        ]
        predictions = tmp_path / f"predictions-{shard}.jsonl"
        write_jsonl(
            predictions,
            [
                {
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "source": row["metadata"]["eval_source"],
                    "protocol": row["metadata"]["eval_protocol"],
                    "delta": None,
                    "target": row["target"],
                    "raw_output": "<label>x</label><diagnosis>x</diagnosis>",
                }
                for row in shard_rows
            ],
        )
        metrics = tmp_path / f"metrics-{shard}.json"
        metrics.write_text(
            json.dumps(
                {
                    "model": "/tmp/model",
                    "adapter": "/mnt/teacher/model",
                    "eval_jsonl": "/tmp/exact.jsonl",
                    "artifact_sha256": {"adapter_model": "a" * 64, "evaluation_input": "b" * 64},
                    "max_samples_per_group": None,
                    "max_length": 1536,
                    "max_new_tokens": 256,
                    "generation_batch_size": 1,
                    "bottleneck_mask": False,
                    "bottleneck_style": None,
                    "unsharded_filtered_rows": 1494,
                    "evaluation_shard": {
                        "index": shard,
                        "count": shard_count,
                    },
                }
            )
        )
        prediction_paths.append(predictions)
        metric_paths.append(metrics)
    return eval_jsonl, prediction_paths, metric_paths


def test_teacher_shard_merge_is_complete_and_strict(tmp_path: Path) -> None:
    eval_jsonl, predictions, metrics = materialize(tmp_path)
    records, summary = merge(predictions, metrics, eval_jsonl)
    assert len(records) == 1494
    assert summary["generation_rows"] == 1494
    assert summary["evaluation_shards"] == 4


def test_teacher_16way_merge_is_complete_and_strict(tmp_path: Path) -> None:
    eval_jsonl, predictions, metrics = materialize(tmp_path, shard_count=16)
    records, summary = merge(predictions, metrics, eval_jsonl)
    assert len(records) == 1494
    assert summary["generation_rows"] == 1494
    assert summary["evaluation_shards"] == 16


def test_teacher_shard_merge_rejects_duplicate(tmp_path: Path) -> None:
    eval_jsonl, predictions, metrics = materialize(tmp_path)
    rows = [json.loads(line) for line in predictions[1].read_text().splitlines()]
    rows[0]["example_id"] = json.loads(predictions[0].read_text().splitlines()[0])["example_id"]
    write_jsonl(predictions[1], rows)
    with pytest.raises(ValueError, match="Duplicate"):
        merge(predictions, metrics, eval_jsonl)
