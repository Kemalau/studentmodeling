from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_stateonly_repair_ab_amlt import validate  # noqa: E402


def config():
    return yaml.safe_load((ROOT / "amlt/feeds_state8_stateonly_repair_ab_20260806.yaml").read_text())


def test_repair_ab_config_passes():
    value = config()
    validate(value)
    assert {job["submit_args"]["env"]["REPAIR_ARM"] for job in value["jobs"]} == {
        "logits", "trajectory01"
    }


def test_repair_ab_rejects_topology_and_arm_drift():
    value = config()
    value["jobs"][0]["sku"] = "80G1-A100"
    with pytest.raises(ValueError, match="Topology drift"):
        validate(value)
    value = config()
    value["jobs"][0]["submit_args"]["env"]["REPAIR_ARM"] = "trajectory01"
    with pytest.raises(ValueError, match="Repair arm drift"):
        validate(value)
