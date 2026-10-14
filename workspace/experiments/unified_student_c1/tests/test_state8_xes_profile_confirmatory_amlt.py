from pathlib import Path

from scripts.validate_state8_xes_profile_confirmatory_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_xes_confirmatory_config_is_bound() -> None:
    validate(ROOT / "amlt/feeds_state8_xes_profile_confirmatory_20260820.yaml")


def test_confirmatory_runner_is_strictly_evaluation_only() -> None:
    source = (ROOT / "scripts/run_state8_xes_profile_confirmatory.sh").read_text()
    assert "state8_profile_ownership_lrfix_v4_formal_world4_seed8801" in source
    assert "state8_profile_ownership_lrfix_v4_formal_seed8801" not in source
    assert "optimizer.step()" not in source
    assert "--init-adapter" not in source
    assert "train_state8_" not in source
    assert "official xes test is touched once" in source.lower()
