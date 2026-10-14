from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_taxonomy_eedi_highcausal_amlt import validate  # noqa: E402


def test_taxonomy_eedi_highcausal_config_is_valid() -> None:
    value = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_taxonomy_eedi_highcausal_20260807.yaml").read_text()
    )
    validate(value)
    env = value["jobs"][0]["submit_args"]["env"]
    assert env["TRAIN_ENTRYPOINT"] == "scripts/train_state8_taxonomy_causal_repair.py"
    assert env["STATE8_EEDI_GEOMETRY_WEIGHT"] == "0.25"
    assert env["STATE8_CAUSAL_WEIGHT"] == "4.0"
