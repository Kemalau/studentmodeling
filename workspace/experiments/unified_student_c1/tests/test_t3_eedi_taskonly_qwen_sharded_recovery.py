from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from merge_t3_eedi_taskonly_shards import merge  # noqa: E402
from validate_t3_eedi_taskonly_qwen_sharded_amlt import validate  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def materialize(tmp_path: Path):
    rows = [
        {
            "example_id": f"eedi-{index:04d}",
            "task": "diagnose_misconception",
            "metadata": {"eval_source": "eedi_mining", "eval_protocol": "open_taxonomy_generation"},
            "target": "<label>x</label><diagnosis>x</diagnosis>",
        }
        for index in range(894)
    ]
    panel = tmp_path / "panel.jsonl"
    write_jsonl(panel, rows)
    predictions, metrics = [], []
    for shard in range(4):
        subset = sorted(rows, key=lambda row: row["example_id"])[shard::4]
        pred = tmp_path / f"pred-{shard}.jsonl"
        write_jsonl(pred, [{"example_id": r["example_id"], "task": r["task"], "source": "eedi_mining", "protocol": "open_taxonomy_generation", "target": r["target"], "raw_output": r["target"]} for r in subset])
        met = tmp_path / f"met-{shard}.json"
        met.write_text(json.dumps({
            "model": "/tmp/model", "adapter": "/mnt/adapter", "eval_jsonl": "/mnt/eval",
            "artifact_sha256": {"adapter_model": "329e95ad02a10cd9541f81ce3f220aaa521369f5761659a146fc2c54ad859bb2", "evaluation_input": "bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8"},
            "max_samples_per_group": None, "max_length": 1536, "max_new_tokens": 160,
            "generation_batch_size": 2, "bottleneck_mask": False, "bottleneck_style": None,
            "unsharded_filtered_rows": 894, "evaluation_shard": {"index": shard, "count": 4},
        }), encoding="utf-8")
        predictions.append(pred); metrics.append(met)
    return panel, predictions, metrics


def test_merge_is_complete_disjoint_and_hash_bound(tmp_path: Path) -> None:
    panel, predictions, metrics = materialize(tmp_path)
    records, summary = merge(predictions, metrics, panel)
    assert len(records) == 894
    assert summary["evaluation_shards"] == 4
    rows = [json.loads(line) for line in predictions[1].read_text().splitlines()]
    rows[0]["example_id"] = json.loads(predictions[0].read_text().splitlines()[0])["example_id"]
    write_jsonl(predictions[1], rows)
    with pytest.raises(ValueError, match="Duplicate"):
        merge(predictions, metrics, panel)


def test_generated_amlt_config_is_compliant() -> None:
    validate(ROOT / "amlt/feeds_t3_eedi_taskonly_qwen_formal_sharded_recovery_20260814.yaml")


def test_watcher_uses_zero_padded_job_selectors() -> None:
    source = (ROOT / "scripts/watch_t3_eedi_taskonly_qwen_sharded_recovery.sh").read_text(encoding="utf-8")
    assert "printf -v suffix '%02d'" in source
    assert 'formal-shard$suffix' in source
