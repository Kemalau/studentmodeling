from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_staged_task_retention_amlt import load, validate  # noqa: E402


def validate_state_path(value: dict) -> None:
    validate(
        value,
        job_name="layerwise-staged-t3-t6-state-path-seed42",
        venv_root="/tmp/studentmodeling-layerwise-staged-statepath-venv",
        loss_profile="state_path",
    )


def test_state_path_config_passes() -> None:
    validate_state_path(load(ROOT / "amlt/feeds_layerwise_staged_state_path_20260802.yaml"))


def test_profile_drift_is_rejected() -> None:
    value = load(ROOT / "amlt/feeds_layerwise_staged_state_path_20260802.yaml")
    value = copy.deepcopy(value)
    value["jobs"][0]["submit_args"]["env"]["LOSS_PROFILE"] = "standard"
    try:
        validate_state_path(value)
    except ValueError as error:
        assert "environment" in str(error)
    else:
        raise AssertionError("loss-profile drift was accepted")
