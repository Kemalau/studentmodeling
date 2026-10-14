from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_layerwise_promoted_checkpoint import audit  # noqa: E402


def test_promoted_manifest_must_match_checkpoint_recipe(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.json"
    manifest.write_text(json.dumps({
        "stage": "layerwise_task_retention_promoted_order_v1",
        "selected_candidate_id": "eedi_full_open_t3__state_path__lr2e-06",
        "selected_loss_profile": "state_path",
        "selected_learning_rate": 2e-6,
        "selected_causal_weight": 2.0,
    }))
    config = {
        "promoted_real_order_applied": True,
        "promoted_real_order_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "promoted_real_order_source_round": 750,
        "promoted_selected_candidate_id": "eedi_full_open_t3__state_path__lr2e-06",
        "promoted_selected_loss_profile": "state_path",
        "promoted_selected_learning_rate": 2e-6,
        "promoted_selected_causal_weight": 2.0,
        "learning_rate": 2e-6,
        "causal_weight": 2.0,
    }
    checkpoint = tmp_path / "recurrent_state.pt"
    torch.save({"distributed": True, "world_size": 16, "round": 1000, "config": config}, checkpoint)
    value = audit(checkpoint, manifest, expected_round=1000)
    assert value["candidate_id"] == "eedi_full_open_t3__state_path__lr2e-06"
    config["causal_weight"] = 1.0
    torch.save({"distributed": True, "world_size": 16, "round": 1000, "config": config}, checkpoint)
    with pytest.raises(ValueError, match="promoted checkpoint recipe drift"):
        audit(checkpoint, manifest, expected_round=1000)
