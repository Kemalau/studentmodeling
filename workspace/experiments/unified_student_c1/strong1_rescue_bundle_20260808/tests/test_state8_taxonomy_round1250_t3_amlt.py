from pathlib import Path

import pytest
import yaml

from scripts.validate_state8_taxonomy_round1250_t3_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_taxonomy_round1250_t3_20260807.yaml"


def test_round1250_t3_config_is_fail_closed() -> None:
    validate(CONFIG)


def test_round1250_t3_validator_allows_only_the_pinned_state_token(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["jobs"][0]["command"] += "\necho <UNRESOLVED_PATH>\n"
    path = tmp_path / "placeholder.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ValueError, match="Unresolved placeholder"):
        validate(path)


def test_round1250_t3_validator_rejects_checkpoint_drift(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
        "checkpoint-round-1250", "checkpoint-round-500"
    )
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ValueError, match="checkpoint-round-1250"):
        validate(path)
