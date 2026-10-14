from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_eedi_open_t3_t6_repair_amlt import load, validate  # noqa: E402


def test_valid_config_and_each_selector() -> None:
    value = load(ROOT / "amlt/feeds_layerwise_eedi_open_t3_t6_repair_20260803.yaml")
    validate(value)
    for job in value["jobs"]:
        one = {**value, "jobs": [job]}
        validate(one, {job["name"]})


def test_accepts_amlt_dump_code_root() -> None:
    value = load(ROOT / "amlt/feeds_layerwise_eedi_open_t3_t6_repair_20260803.yaml")
    value["code"].pop("local_dir")
    value["code"]["storage_id"] = "default"
    validate(value)


def test_rejects_unrecognized_dump_code_root() -> None:
    value = load(ROOT / "amlt/feeds_layerwise_eedi_open_t3_t6_repair_20260803.yaml")
    value["code"].pop("local_dir")
    value["code"]["storage_id"] = "other"
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("LEARNING_RATE", "5e-6"),
        ("WARM_START_EXPECTED_ROUND", "750"),
        ("DIAGNOSTIC_START_GLOBAL_POSITION", "12000"),
        ("_AZUREML_SINGULARITY_JOB_UAI", "wrong"),
    ],
)
def test_rejects_scientific_or_infrastructure_drift(field: str, bad: str) -> None:
    value = load(ROOT / "amlt/feeds_layerwise_eedi_open_t3_t6_repair_20260803.yaml")
    changed = copy.deepcopy(value)
    changed["jobs"][0]["submit_args"]["env"][field] = bad
    with pytest.raises(ValueError):
        validate(changed)
