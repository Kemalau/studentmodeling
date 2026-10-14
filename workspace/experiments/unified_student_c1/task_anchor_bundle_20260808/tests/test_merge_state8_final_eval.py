from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from merge_state8_final_eval import merge_formal  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_merge_rejects_duplicate_shard_coordinates(tmp_path: Path) -> None:
    metrics = []
    predictions = []
    for index in range(4):
        metric = tmp_path / f"m{index}.json"
        write_json(
            metric,
            {
                "evaluation_shard": {"index": 0, "count": 4},
                "model": "m",
                "adapter": "a",
                "recurrent_state": "s",
                "state_architecture": "layerwise",
                "state_shape": [64, 8, 5120],
                "raw_history_visible_to_decoder": False,
                "persistent_history_kv_cache": False,
                "unsharded_filtered_rows": 21676,
                "sampled_rows": 5419,
                "artifact_sha256": {"adapter_model": "a", "recurrent_state": "s"},
            },
        )
        prediction = tmp_path / f"p{index}.jsonl"
        prediction.write_text("", encoding="utf-8")
        metrics.append(metric)
        predictions.append(prediction)
    panel = tmp_path / "eval.jsonl"
    panel.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate"):
        merge_formal(predictions, metrics, panel)


def test_source_explicitly_requires_closed_state8() -> None:
    source = (ROOT / "scripts/merge_state8_final_eval.py").read_text()
    assert "[64, 8, 5120]" in source
    assert "raw_history_visible_to_decoder" in source
    assert "persistent_history_kv_cache" in source
    assert "Duplicate example IDs" in source


def test_final_gate_uses_semantics_not_t3_exact_match() -> None:
    source = (ROOT / "scripts/validate_state8_final_eval.py").read_text()
    assert "semantic_no_exact_shortcuts" in source
    assert '"semantic_backend": judge.get("judge_backend") == "trapi"' in source
    assert '"T3_exact_match_role": "auxiliary_only_not_used_for_gate"' in source
    assert '"final_fit_artifact_binding"' in source
