from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_equal_state_kd_amlt import load, validate  # noqa: E402


def test_equal_state_kd_is_a_matched_closed_state_arm() -> None:
    path = ROOT / "amlt/feeds_layerwise_equal_state_kd_20260802.yaml"
    validate(load(path))
    text = path.read_text(encoding="utf-8")
    assert "BRANCH: equal_t3_t6" in text
    assert "LOSS_PROFILE: state_kd" in text
    assert "80G4-A100" in text
    assert "rankfun_aml" in text
    assert "layerwise_task_retention_equal_t3_t6_seed42_v1.json" in text

