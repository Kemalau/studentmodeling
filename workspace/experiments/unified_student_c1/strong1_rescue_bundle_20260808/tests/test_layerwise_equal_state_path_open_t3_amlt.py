from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_equal_state_path_open_t3_amlt import load, validate  # noqa: E402


def config() -> dict:
    return load(ROOT / "amlt/feeds_layerwise_equal_state_path_open_t3_20260803.yaml")


def test_equal_open_state_path_is_a_matched_closed_state_arm() -> None:
    value = config()
    validate(value)
    env = value["jobs"][0]["submit_args"]["env"]
    assert env["BRANCH"] == "equal_t3_t6_open_t3"
    assert env["LOSS_PROFILE"] == "state_path"
    assert env["TRAIN_ENTRYPOINT"] == "scripts/train_recurrent_student_m4_open_t3.py"


@pytest.mark.parametrize(
    ("key", "bad"),
    (("LOSS_PROFILE", "state_kd"), ("BRANCH", "equal_t3_t6"), ("_AZUREML_SINGULARITY_JOB_UAI", "other")),
)
def test_scientific_or_identity_drift_fails_closed(key: str, bad: str) -> None:
    value = copy.deepcopy(config())
    value["jobs"][0]["submit_args"]["env"][key] = bad
    with pytest.raises(ValueError):
        validate(value)
