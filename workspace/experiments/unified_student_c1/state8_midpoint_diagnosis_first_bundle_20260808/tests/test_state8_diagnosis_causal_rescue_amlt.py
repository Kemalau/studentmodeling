from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_diagnosis_causal_rescue_amlt import load, validate  # noqa: E402


def config() -> dict:
    return load(ROOT / "amlt/feeds_state8_diagnosis_causal_rescue_20260806.yaml")


def test_frozen_config_passes() -> None:
    validate(config())


def test_rejects_wrong_identity() -> None:
    value = copy.deepcopy(config())
    value["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
    with pytest.raises(ValueError, match="UAI"):
        validate(value)


def test_rejects_single_gpu_training() -> None:
    value = config()
    value["jobs"][0]["sku"] = "80G1-A100"
    with pytest.raises(ValueError, match="Topology"):
        validate(value)


def test_launcher_resumes_from_latest_atomic_checkpoint() -> None:
    source = (ROOT / "scripts/run_state8_diagnosis_causal_rescue.sh").read_text()
    for fragment in (
        "select_latest_valid_distributed_checkpoint.py",
        "stage_distributed_resume_bundle.py",
        "--expected-world-size 4 --max-round 250",
        "--save-steps 25 --recovery-save-steps 25",
        "--probe-expected-cadence",
        "resume_checkpoint_selected=",
        '--resume-from-checkpoint "$resume_checkpoint"',
    ):
        assert fragment in source
