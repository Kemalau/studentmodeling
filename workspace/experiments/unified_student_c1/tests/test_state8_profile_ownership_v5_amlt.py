from pathlib import Path

import pytest
import yaml

from scripts.validate_state8_profile_ownership_v5_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "amlt/feeds_state8_profile_ownership_v5_smoke_20260821.yaml"
FORMAL = ROOT / "amlt/feeds_state8_profile_ownership_v5_formal_20260821.yaml"


def test_v5_configs_are_bound() -> None:
    validate(SMOKE, "smoke")
    validate(FORMAL, "formal")


def test_v5_rejects_loss_budget_drift(tmp_path: Path) -> None:
    value = yaml.safe_load(SMOKE.read_text())
    value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
        "STATE8_V5_PROFILE_GENERATION_WEIGHT=0.1",
        "STATE8_V5_PROFILE_GENERATION_WEIGHT=1.0",
    )
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="command binding drift"):
        validate(path, "smoke")


def test_v5_rejects_formal_diagnostic_leak(tmp_path: Path) -> None:
    value = yaml.safe_load(FORMAL.read_text())
    value["jobs"][0]["command"] += "\nexport PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS=200\n"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="accidentally bounded"):
        validate(path, "formal")
