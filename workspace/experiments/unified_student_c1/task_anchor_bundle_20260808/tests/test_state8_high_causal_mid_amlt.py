from pathlib import Path

import yaml

from scripts.validate_state8_high_causal_mid_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_mid_config_is_matched_and_valid() -> None:
    value = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_high_causal_mid_20260806.yaml").read_text()
    )
    validate(value)
    env = value["jobs"][0]["submit_args"]["env"]
    assert env["STATE8_CAUSAL_WEIGHT"] == "4.0"
    assert env["STATE8_GEOMETRY_WEIGHT"] == "0.5"
    assert env["RUN_TAG"] == "highcausal_mid_v1"
