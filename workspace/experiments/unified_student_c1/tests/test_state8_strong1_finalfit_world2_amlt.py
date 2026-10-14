from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.materialize_state8_strong1_finalfit_world2_amlt import materialize
from scripts.validate_state8_strong1_finalfit_world2_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_materialized_world2_config_is_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "world2.yaml"
    materialize(ROOT / "amlt/feeds_state8_strong1_widen8_final_fit_20260808.yaml", output)
    config = yaml.safe_load(output.read_text())
    assert len(config["jobs"]) == 10
    assert all(job["sku"] == "80G2-A100" for job in config["jobs"])
    assert all(
        "run_state8_unified_final_fit_world2.sh" in str(job["command"])
        for job in config["jobs"]
    )
    validate(config)


def test_world2_validator_rejects_topology_or_identity_drift(tmp_path: Path) -> None:
    output = tmp_path / "world2.yaml"
    materialize(ROOT / "amlt/feeds_state8_strong1_widen8_final_fit_20260808.yaml", output)
    config = yaml.safe_load(output.read_text())
    bad = copy.deepcopy(config)
    bad["jobs"][0]["sku"] = "80G1-A100"
    with pytest.raises(ValueError, match="Topology drift"):
        validate(bad)
    bad = copy.deepcopy(config)
    bad["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
    with pytest.raises(ValueError, match="Environment/UAI/variant drift"):
        validate(bad)


def test_checked_in_world2_config_matches_every_world4_variant() -> None:
    world4 = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_strong1_widen8_final_fit_20260808.yaml").read_text()
    )
    world2 = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_strong1_widen8_final_fit_world2_20260808.yaml").read_text()
    )
    coordinates4 = {
        (job["name"], job["submit_args"]["env"]["FINALFIT_VARIANT"])
        for job in world4["jobs"]
    }
    coordinates2 = {
        (job["name"], job["submit_args"]["env"]["FINALFIT_VARIANT"])
        for job in world2["jobs"]
    }
    assert coordinates2 == coordinates4
    validate(world2)
