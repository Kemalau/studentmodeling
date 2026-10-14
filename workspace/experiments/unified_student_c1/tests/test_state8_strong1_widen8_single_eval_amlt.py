from pathlib import Path

from scripts.validate_state8_strong1_widen8_single_eval_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_single_fallback_eval_is_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_strong1_widen8_single_eval_20260808.yaml")


def test_single_fallback_is_scored_by_joint_bridge_selector() -> None:
    source = (ROOT / "scripts/select_state8_strong1_widen8_bridge.py").read_text()
    assert '"single_lr5e7_seed60"' in source
    assert '"state8-strong1-widen8-single-eval-lr5e7-seed60"' in source
