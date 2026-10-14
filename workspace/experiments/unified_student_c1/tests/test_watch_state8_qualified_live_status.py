from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "scripts/watch_state8_qualified_live_status.sh").read_text()


def test_live_status_watcher_is_locked_periodic_and_fail_closed() -> None:
    assert "flock -n 9" in TEXT
    assert "summarize_state8_qualified_pipeline.py" in TEXT
    assert "state8_qualified_pipeline_live_status.json" in TEXT
    assert 'touch "$state/needs_attention"' in TEXT
    assert "sleep 60" in TEXT
    assert "controller.heartbeat" in TEXT
