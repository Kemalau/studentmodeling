from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_highcausal_progress_watchdog_covers_both_arms_and_health() -> None:
    text = (ROOT / "scripts/watch_state8_highcausal_training_progress.sh").read_text()
    for fragment in (
        "state8-highcausal-joint-seed47",
        "state8-highcausal-updater-seed48",
        "state8-highcausal-mid-seed49",
        "state8-highcausal-eedi-seed50",
        "state8-taxonomy-eedi-highcausal-seed51",
        "monitor_layerwise_unified_training_progress.py",
        "latest_rank0.log",
        "seconds_since_progress",
        "health",
        "stalled.json",
        "--total-rounds 250",
        "STALL_SECONDS:-3600",
    ):
        assert fragment in text
    assert "amlt cancel" not in text
    assert ".azure az account get-access-token" in text
    assert ".azure-keyflip-feeds" in text
