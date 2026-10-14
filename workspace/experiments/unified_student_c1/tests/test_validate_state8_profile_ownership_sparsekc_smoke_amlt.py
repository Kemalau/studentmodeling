from pathlib import Path

import pytest
import yaml

from scripts.validate_state8_profile_ownership_sparsekc_smoke_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt" / "feeds_state8_profile_ownership_lrfix_smoke_20260820.yaml"


def test_sparsekc_smoke_config_is_bound() -> None:
    validate(CONFIG)


def test_sparsekc_smoke_rejects_support_threshold_drift(tmp_path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
        "STATE8_PROFILE_OWNERSHIP_MIN_KC_TRAIN_SUPPORT=4",
        "STATE8_PROFILE_OWNERSHIP_MIN_KC_TRAIN_SUPPORT=1",
    )
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="command binding drift"):
        validate(path)
