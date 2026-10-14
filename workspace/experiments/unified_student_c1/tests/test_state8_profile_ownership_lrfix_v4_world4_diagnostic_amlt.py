from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.validate_state8_profile_ownership_lrfix_v4_world4_diagnostic_amlt import (
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_profile_ownership_lrfix_v4_world4_diagnostic_20260821.yaml"
RUNNER = ROOT / "scripts/run_state8_student_profile_readout.sh"


def test_exact_v4_world4_diagnostic_passes() -> None:
    validate(CONFIG)
    assert hashlib.sha256(RUNNER.read_bytes()).hexdigest() in CONFIG.read_text()


def test_diagnostic_rejects_unbounded_training(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
        "export PROFILE_READOUT_DIAGNOSTIC_STOP_ROUND=2\n", ""
    )
    path = tmp_path / "unbounded.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="command binding drift"):
        validate(path)


def test_diagnostic_rejects_single_rank_drift(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["jobs"][0]["sku"] = "80G1-A100"
    value["jobs"][0]["process_count_per_node"] = 1
    value["jobs"][0]["mpi"] = False
    path = tmp_path / "single.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="Topology drift"):
        validate(path)
