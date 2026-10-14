from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_teacher_stage_a_amlt import load, validate  # noqa: E402


def config() -> dict:
    return load(ROOT / "amlt/feeds_state8_teacher_stage_a_20260805.yaml")


def test_teacher_stage_a_config_passes() -> None:
    validate(config())


def test_teacher_stage_a_rejects_topology_or_uai_drift() -> None:
    value = config()
    value["jobs"][0]["sku"] = "G1"
    with pytest.raises(ValueError, match="Topology"):
        validate(value)


def test_teacher_stage_a_is_single_process_to_avoid_shared_output_corruption() -> None:
    value = config()["jobs"][0]
    assert value["sku"] == "80G1-A100"
    assert value["process_count_per_node"] == 1
    assert value["mpi"] is False
    assert "--gradient-accumulation-steps 16" in value["command"]
    value = config()
    value["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
    with pytest.raises(ValueError, match="UAI"):
        validate(value)


def test_teacher_stage_a_rejects_loss_of_exact_epoch() -> None:
    value = copy.deepcopy(config())
    value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
        "--max-steps -1 --num-train-epochs 1", "--max-steps 100"
    )
    with pytest.raises(ValueError, match="launcher drift"):
        validate(value)
