from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_semantic_causal_screen_amlt import validate  # noqa: E402


def config() -> dict:
    return yaml.safe_load(
        (ROOT / "amlt/feeds_state8_semantic_causal_screen_20260806.yaml").read_text()
    )


def test_semantic_causal_config_passes() -> None:
    validate(config())


def test_semantic_causal_config_rejects_objective_drift() -> None:
    value = copy.deepcopy(config())
    value["jobs"][0]["submit_args"]["env"]["STATE8_RANDOM_NEUTRAL"] = "0"
    with pytest.raises(ValueError, match="Environment drift"):
        validate(value)


def test_semantic_causal_config_rejects_input_checkpoint_drift() -> None:
    value = copy.deepcopy(config())
    value["jobs"][0]["submit_args"]["env"]["STUDENT_ROOT"] = "/tmp/model"
    with pytest.raises(ValueError, match="Environment drift"):
        validate(value)


def test_geometry_grid_config_passes() -> None:
    value = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_semantic_causal_grid_20260806.yaml").read_text()
    )
    validate(value)
    assert {
        job["submit_args"]["env"]["STATE8_GEOMETRY_WEIGHT"]
        for job in value["jobs"]
    } == {"0.05", "0.25"}


def test_eedi_geometry_grid_config_passes() -> None:
    value = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_eedi_semantic_geometry_grid_20260806.yaml").read_text()
    )
    validate(value)
    assert {
        job["submit_args"]["env"]["STATE8_EEDI_GEOMETRY_WEIGHT"]
        for job in value["jobs"]
    } == {"0.05", "0.10"}


def test_eedi_high_geometry_grid_config_passes() -> None:
    value = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_eedi_semantic_geometry_high_grid_20260806.yaml").read_text()
    )
    validate(value)
    assert {
        job["submit_args"]["env"]["STATE8_EEDI_GEOMETRY_WEIGHT"]
        for job in value["jobs"]
    } == {"0.25", "0.50"}
