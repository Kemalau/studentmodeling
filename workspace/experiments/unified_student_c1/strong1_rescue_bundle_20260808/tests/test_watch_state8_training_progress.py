from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "scripts/watch_state8_training_progress.sh").read_text()


def test_state8_progress_watchdog_covers_all_arms_and_preserves_stall_clock() -> None:
    for fragment in (
        "primary",
        "balanced",
        "highkd",
        "repair_trajectory01",
        "repair_logits",
        "state8-repair-trajectory01-seed44",
        "state8-repair-logits-seed44",
        "total_rounds=250",
        "STALL_SECONDS:-3600",
        "monitor_layerwise_unified_training_progress.py",
        "--awaiting-first-record",
        "std_log_process_(?:0|00)",
        "seconds_since_progress",
        "stalled.json",
        "health_alert.json",
        '[[ "$status_value" != running ]]',
    ):
        assert fragment in TEXT
    assert "amlt cancel" not in TEXT
    assert "amlt run" not in TEXT
