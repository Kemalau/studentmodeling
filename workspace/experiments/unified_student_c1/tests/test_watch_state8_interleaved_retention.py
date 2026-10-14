from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watcher_accepts_recovery_binding_and_uses_current_results_cli() -> None:
    text = (ROOT / "scripts/watch_state8_interleaved_retention_recovery1.sh").read_text(
        encoding="utf-8"
    )
    assert 'STATE8_INTERLEAVED_EXPERIMENT:-state8-interleaved-retention-recovery1-20260813' in text
    assert 'STATE8_INTERLEAVED_JOB_ID:-blue_nut_db3kvbymhg' in text
    assert 'STATE8_INTERLEAVED_STATE_DIR:-results/state8_interleaved_retention_recovery1_20260813' in text
    assert 'amlt results download "$experiment" --output "$state_dir/download"' in text
    assert 'amlt results "$experiment"' not in text
    assert text.count("source scripts/use_studentmodeling_feeds_auth.sh") == 1
    assert "AUTH_OR_STATUS_RETRY" in text
