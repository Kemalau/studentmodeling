from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from publish_state8_final_checkpoint import STAGE, publish  # noqa: E402


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source(
    root: Path,
    variant: str = "t3_heavy",
    adapter: bytes = b"adapter",
    history_mode: str = "five_stage_curriculum",
) -> Path:
    root.mkdir()
    (root / "final_adapter").mkdir()
    state = b"state"
    (root / "final_adapter/adapter_model.safetensors").write_bytes(adapter)
    (root / "final_adapter/adapter_config.json").write_text("{}\n")
    (root / "recurrent_state.pt").write_bytes(state)
    completion = {
        "status": "complete",
        "micro_steps": 259845,
        "exact_history_support": True,
        "sha256": {"adapter_model": sha(adapter), "recurrent_state": sha(state)},
    }
    receipt = {
        "stage": "state8_qualified_unified_final_fit_v1",
        "status": "final_fit_complete",
        "one_epoch_exact": True,
        "selected_replay_variant": variant,
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "adapter_sha256": sha(adapter),
        "recurrent_state_sha256": sha(state),
        "history_training_mode": history_mode,
    }
    if history_mode == "strict_state_only":
        receipt.update(
            {
                "history_dropout_counts": {"1.0": 259845},
                "online_kd_weight": 1.0,
                "label_auxiliary_weight": 4.0,
            }
        )
    (root / "completion.json").write_text(json.dumps(completion))
    (root / "state8_final_fit_receipt.json").write_text(json.dumps(receipt))
    for name in (
        "run_config.json",
        "training_metrics.json",
        "validation_metrics.json",
        "causal_validation_metrics.json",
    ):
        (root / name).write_text("{}\n")
    view = (
        "state8_stateonly_teacher_repair.json"
        if history_mode == "strict_state_only"
        else "state8_external_teacher_curriculum.json"
    )
    (root / view).write_text("{}\n")
    if variant == "highcausal_taxonomy":
        (root / "taxonomy_causal_repair.json").write_text("{}\n")
    return root


def test_publish_is_complete_and_idempotent(tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    dst = tmp_path / "canonical"
    first = publish(src, dst)
    second = publish(src, dst)
    assert first == second
    assert first["stage"] == STAGE
    assert first["status"] == "complete"
    assert first["selected_replay_variant"] == "t3_heavy"
    assert set(first["payload_sha256"]) == {
        "final_adapter/adapter_model.safetensors",
        "recurrent_state.pt",
        "completion.json",
        "state8_final_fit_receipt.json",
        "run_config.json",
        "training_metrics.json",
        "validation_metrics.json",
        "causal_validation_metrics.json",
        "state8_external_teacher_curriculum.json",
    }
    assert (dst / "final_adapter/adapter_model.safetensors").read_bytes() == b"adapter"
    assert not (dst / "canonical_publish_receipt.json.partial").exists()
    assert not list(tmp_path.glob("canonical.staging-*"))


def test_publish_rejects_conflicting_existing_destination(tmp_path: Path) -> None:
    first = source(tmp_path / "first")
    second = source(tmp_path / "second", variant="balanced", adapter=b"different")
    dst = tmp_path / "canonical"
    publish(first, dst)
    with pytest.raises(ValueError, match="different artifacts"):
        publish(second, dst)


def test_publish_accepts_strict_repair_variant(tmp_path: Path) -> None:
    src = source(
        tmp_path / "source",
        variant="repair_logits",
        history_mode="strict_state_only",
    )
    dst = tmp_path / "canonical"
    receipt = publish(src, dst)
    assert receipt["selected_replay_variant"] == "repair_logits"
    assert (dst / "state8_stateonly_teacher_repair.json").is_file()


def test_publish_accepts_highcausal_mid_variant(tmp_path: Path) -> None:
    src = source(
        tmp_path / "source",
        variant="highcausal_mid",
        history_mode="strict_state_only",
    )
    receipt = publish(src, tmp_path / "canonical")
    assert receipt["selected_replay_variant"] == "highcausal_mid"


def test_publish_accepts_highcausal_eedi_variant(tmp_path: Path) -> None:
    src = source(
        tmp_path / "source",
        variant="highcausal_eedi",
        history_mode="strict_state_only",
    )
    receipt = publish(src, tmp_path / "canonical")
    assert receipt["selected_replay_variant"] == "highcausal_eedi"


def test_publish_accepts_highcausal_taxonomy_variant(tmp_path: Path) -> None:
    src = source(
        tmp_path / "source",
        variant="highcausal_taxonomy",
        history_mode="strict_state_only",
    )
    receipt = publish(src, tmp_path / "canonical")
    assert receipt["selected_replay_variant"] == "highcausal_taxonomy"
    assert (tmp_path / "canonical/taxonomy_causal_repair.json").is_file()


def test_publish_recovers_after_payload_rename_before_receipt(tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    dst = tmp_path / "canonical"
    publish(src, dst)
    (dst / "canonical_publish_receipt.json").unlink()
    recovered = publish(src, dst)
    assert recovered["stage"] == STAGE
    assert (dst / "canonical_publish_receipt.json").is_file()


def test_publish_rejects_incomplete_stranded_destination(tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    dst = tmp_path / "canonical"
    dst.mkdir()
    with pytest.raises(ValueError, match="Incomplete canonical destination"):
        publish(src, dst)


def test_publish_rejects_missing_metadata_from_published_destination(tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    dst = tmp_path / "canonical"
    publish(src, dst)
    (dst / "validation_metrics.json").unlink()
    with pytest.raises(ValueError, match="manifest is incomplete"):
        publish(src, dst)
