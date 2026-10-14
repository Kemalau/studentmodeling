from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_layerwise_staged_task_retention.sh").read_text()


def test_monitor_recovers_and_downloads_complete_scientific_artifacts() -> None:
    for fragment in (
        "validate_layerwise_staged_task_retention_amlt.py",
        "recurrent_submit_guard",
        "amlt results download",
        "eval/c4/recurrent_causal_gate.json",
        "eval/t6_full/metrics.json",
        "judge_formal_t3_predictions.py",
        "validate_checkpoint_t3_semantic_result.py",
        "summarize_t3_seen_unseen.py",
        "seen_unseen.json",
        "summarize_layerwise_single_task_retention.py",
    ):
        assert fragment in SOURCE


def test_monitor_has_bounded_recovery_and_heartbeats() -> None:
    assert "heartbeat" in SOURCE
    assert "wave < 5" in SOURCE
    assert "needs_attention" in SOURCE
