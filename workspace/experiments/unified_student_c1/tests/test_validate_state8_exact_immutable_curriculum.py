from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_state8_exact_immutable_curriculum import EXPECTED_STAGE, validate


def put(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_bytes(value)


def fixture(root: Path) -> Path:
    adapter = b"adapter"
    state = b"state"
    artifacts = {
        "adapter_model": hashlib.sha256(adapter).hexdigest(),
        "recurrent_state": hashlib.sha256(state).hexdigest(),
    }
    put(root / "model/adapter_model.safetensors", adapter)
    put(root / "model/adapter_config.json", {})
    put(root / "model/recurrent_state.pt", state)
    put(
        root / "metrics/immutable_receipt.json",
        {
            "status": "complete",
            "stage": EXPECTED_STAGE,
            "replay_variant": "t3_heavy",
            "state_shape": [64, 8, 5120],
            "micro_steps": 5000,
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "semantic_judge_authoritative": True,
            "exact_label_metric_authoritative": False,
            "diagnosis_field_order": ["diagnosis", "evidence", "label"],
            "canonical_label_role": "auxiliary_ordinary_weight",
            "label_auxiliary_weight": 1.0,
            "formal_screen_rows": 487,
            "sha256": artifacts,
        },
    )
    put(
        root / "metrics/completion.json",
        {
            "status": "complete",
            "diagnostic": False,
            "micro_steps": 5000,
            "real_examples_seen_once": True,
            "all_causal_pairs_covered": True,
            "causal_exposure_count_matches_plan": True,
            "sha256": artifacts,
        },
    )
    put(
        root / "metrics/qualified_teacher_receipt.json",
        {
            "status": "complete",
            "stage": "state8_qualified_teacher_curriculum_v1",
            "history_dropout_counts": {
                "0.0": 1000,
                "0.25": 1000,
                "0.5": 1000,
                "0.75": 1000,
                "1.0": 1000,
            },
            "state_shape": [64, 8, 5120],
            "raw_history_visible_to_decoder": False,
            "semantic_judge_authoritative": True,
            "exact_label_metric_authoritative": False,
            "adapter_sha256": artifacts["adapter_model"],
            "recurrent_state_sha256": artifacts["recurrent_state"],
        },
    )
    put(
        root / "metrics/diagnosis_first_teacher_curriculum.json",
        {
            "status": "complete",
            "stage": "state8_diagnosis_first_teacher_curriculum_v1",
            "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
            "field_order": ["diagnosis", "evidence", "label"],
            "semantic_fields_primary": True,
            "canonical_label_role": "auxiliary_ordinary_weight",
            "semantic_judge_authoritative": True,
            "exact_label_metric_authoritative": False,
            "raw_history_visible_to_decoder": False,
            "final_inference_route": "strict_state_only",
        },
    )
    put(
        root / "eval/formal_screen/metrics.json",
        {
            "artifact_sha256": artifacts,
            "state_shape": [64, 8, 5120],
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "unsharded_filtered_rows": 21676,
            "sampled_rows": 487,
        },
    )
    predictions = "".join(
        json.dumps({"example_id": f"example-{index}"}) + "\n" for index in range(487)
    )
    path = root / "eval/formal_screen/predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(predictions, encoding="utf-8")
    return root


def test_complete_immutable_curriculum_is_reusable(tmp_path: Path) -> None:
    result = validate(fixture(tmp_path))
    assert result["status"] == "complete"
    assert result["formal_screen_rows"] == 487
    assert all(result["checks"].values())


def test_artifact_drift_fails_closed(tmp_path: Path) -> None:
    root = fixture(tmp_path)
    (root / "model/recurrent_state.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact_hashes"):
        validate(root)


def test_incomplete_predictions_fail_closed(tmp_path: Path) -> None:
    root = fixture(tmp_path)
    path = root / "eval/formal_screen/predictions.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="formal_predictions_complete"):
        validate(root)
