from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_finalfit_milestone_eval_amlt import load, validate  # noqa: E402


def config() -> dict:
    return load(ROOT / "amlt/feeds_state8_finalfit_milestone_eval_20260805.yaml")


def test_milestone_config_passes() -> None:
    value = config(); validate(value); assert len(value["jobs"]) == 4


def test_single_milestone_recovery_passes() -> None:
    value = config(); value["jobs"] = [value["jobs"][1]]
    validate(value, {"state8-finalfit-milestone-r6497"})


def test_milestone_rejects_training_topology() -> None:
    value = copy.deepcopy(config()); value["jobs"][0]["sku"] = "80G4-A100"
    with pytest.raises(ValueError, match="Topology"):
        validate(value)
