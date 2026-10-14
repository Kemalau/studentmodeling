from pathlib import Path

import yaml

from scripts.validate_state8_source_anchor_checkpoint100_screen_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_source_anchor_checkpoint100_screen_20260811.yaml"


def test_checkpoint100_config_is_valid() -> None:
    validate(CONFIG)


def test_checkpoint100_config_supports_recovery_subset(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["jobs"] = [value["jobs"][1]]
    path = tmp_path / "subset.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    validate(path, {"state8-source-anchor-cp100-updater-mse1"})


def test_checkpoint100_validator_rejects_partial_hash(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["jobs"][0]["submit_args"]["env"]["SOURCE_ANCHOR_STATE_SHA256"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    try:
        validate(path)
    except ValueError as error:
        assert "Environment" in str(error)
    else:
        raise AssertionError("Wrong checkpoint hash was accepted")
