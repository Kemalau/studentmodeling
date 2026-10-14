from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_semantic_progress_watchdog_covers_all_candidates_and_stalls() -> None:
    text = (ROOT / "scripts/watch_state8_semantic_training_progress.sh").read_text()
    for fragment in (
        "c4_g005",
        "c4_g010",
        "c4_g025",
        "eedi_g005",
        "eedi_g010",
        "eedi_g025",
        "eedi_g050",
        "eedi_g025_single",
        "monitor_layerwise_unified_training_progress.py",
        "latest_rank0.log",
        "seconds_since_progress",
        "stalled.json",
        "total_rounds=1000",
        "all_selection_arms_complete",
        'touch "$controller/complete"',
    ):
        assert fragment in text
