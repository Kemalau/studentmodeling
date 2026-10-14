from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_high_causal_eedi_amlt import validate  # noqa: E402


def test_eedi_highcausal_config_is_strict_state_only_and_valid() -> None:
    value = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_high_causal_eedi_20260807.yaml").read_text()
    )
    validate(value)
    env = value["jobs"][0]["submit_args"]["env"]
    assert env["STATE8_CAUSAL_WEIGHT"] == "4.0"
    assert env["STATE8_GEOMETRY_WEIGHT"] == "0.5"
    assert env["STATE8_EEDI_GEOMETRY_WEIGHT"] == "0.25"
    assert env["RUN_TAG"] == "highcausal_eedi025_v1"
