from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generic_watcher_reauthenticates_and_collects_terminal_artifacts() -> None:
    text = (ROOT / "scripts/watch_amlt_job_resilient.sh").read_text(encoding="utf-8")
    assert "source scripts/use_studentmodeling_feeds_auth.sh" in text
    assert "AUTH_OR_STATUS_RETRY" in text
    assert 'timeout 60 az ml job show' in text
    assert 'timeout --foreground 600 amlt results download "$WATCH_EXPERIMENT" --output "$WATCH_STATE_DIR/download"' in text
    assert 'timeout 180 amlt logs view -n 400 "$WATCH_EXPERIMENT"' in text
    assert "STUDENTMODELING_FEEDS_AUTH_HEALTH" in text
    assert 'touch "$WATCH_STATE_DIR/remote_passed"' in text
    assert text.index('timeout --foreground 600 amlt results download "$WATCH_EXPERIMENT"') < text.index(
        'touch "$WATCH_STATE_DIR/remote_passed"'
    )
