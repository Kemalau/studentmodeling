from pathlib import Path

from scripts.validate_state8_strong1_widen8_eval_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_strong1_widen8_eval_is_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_strong1_widen8_eval_20260808.yaml")

