from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_layerwise_equal_state_path_open_t3.sh").read_text()


def test_open_t3_state_path_monitor_reuses_audited_recovery_monitor() -> None:
    for fragment in (
        "STATE_SLUG=equal_state_path_open_t3",
        "feeds_layerwise_equal_state_path_open_t3_20260803.yaml",
        "BASE_EXPERIMENT=layerwise-equal-t3-t6-open-t3-state-path-seed42-20260803",
        "LOSS_PROFILE_EXPECTED=state_path",
        "SUMMARY_BRANCH_EXPECTED=equal_t3_t6_open_t3",
        "validate_layerwise_equal_state_path_open_t3_amlt.py",
        "exec bash scripts/watch_layerwise_equal_state_path.sh",
    ):
        assert fragment in SOURCE
