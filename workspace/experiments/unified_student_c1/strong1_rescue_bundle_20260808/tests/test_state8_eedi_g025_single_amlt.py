import copy
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_eedi_g025_single_amlt import validate  # noqa: E402


def config():
    return yaml.safe_load(
        (ROOT / "amlt/feeds_state8_eedi_g025_single_a100_20260806.yaml").read_text()
    )


def test_single_config_passes() -> None:
    validate(config())


def test_single_a100_matches_four_gpu_optimizer_exposure() -> None:
    value = config()
    env = value["jobs"][0]["submit_args"]["env"]
    assert env["REQUESTED_WORLD_SIZE"] == "1"
    assert env["GRADIENT_ACCUMULATION_STEPS"] == "4"
    launcher = (ROOT / "scripts/run_state8_stateonly_repair_ab.sh").read_text()
    assert '--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"' in launcher
    assert "assert c['optimizer_steps']==250" in launcher


def test_single_config_rejects_topology_drift() -> None:
    value = copy.deepcopy(config())
    value["jobs"][0]["sku"] = "80G4-A100"
    with pytest.raises(ValueError, match="Topology drift"):
        validate(value)
