from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_state8_finalfit_milestone import summarize  # noqa: E402


def write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def inputs(root: Path, *, persistent: bool = False) -> tuple[Path, Path, Path, Path]:
    artifacts = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    receipt = write(root / "receipt.json", {
        "stage": "state8_finalfit_milestone_eval_v1", "status": "complete",
        "variant": "t3_heavy", "round": 3249,
        "source_checkpoint_sha256": artifacts,
        "evaluated_artifact_sha256": artifacts,
    })
    formal = write(root / "formal.json", {
        "stage": "layerwise_finalfit_checkpoint3249_formal_screen",
        "checkpoint_round": 3249, "diagnostic_gate_pass": True,
    })
    semantic = write(root / "semantic.json", {
        "stage": "layerwise_finalfit_checkpoint3249_T3_semantic_judge",
        "checkpoint_round": 3249, "pass": True,
    })
    c4 = write(root / "c4.json", {
        "pair_seed": 42, "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": persistent,
        "artifact_sha256": artifacts,
        "summary": {"clean_n": 80, "D_specific": 0.2, "scientific_gate_pass": True},
    })
    return receipt, formal, semantic, c4


def test_joint_milestone_passes(tmp_path: Path) -> None:
    value = summarize(*inputs(tmp_path))
    assert value["pass"] is True
    assert all(value["checks"].values())


def test_joint_milestone_rejects_persistent_history(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="persists history"):
        summarize(*inputs(tmp_path, persistent=True))
