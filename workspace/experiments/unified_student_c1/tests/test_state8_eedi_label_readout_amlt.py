from pathlib import Path

from scripts.validate_state8_eedi_label_readout_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_label_readout_amlt_is_bound_and_read_only():
    validate(ROOT / "amlt/feeds_state8_eedi_label_readout_20260813.yaml")
