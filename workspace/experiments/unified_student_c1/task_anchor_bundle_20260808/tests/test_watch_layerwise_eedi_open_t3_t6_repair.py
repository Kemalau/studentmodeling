from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_layerwise_eedi_open_t3_t6_repair.sh").read_text()
SUPERVISOR = (ROOT / "scripts/supervise_recurrent_unified_pipeline.sh").read_text()


def test_watcher_recovers_each_arm_and_runs_frozen_gates() -> None:
    for fragment in (
        "submit_recovery",
        "waves[\"$job\"] <= 6",
        "validate_layerwise_eedi_open_t3_t6_repair_amlt.py",
        "passed_receipts",
        "retired_arms",
        "selection_complete",
        "passed_count == 0",
        "retired without a selectable scientific result",
        "judge_formal_t3_predictions.py",
        "summarize_t3_seen_unseen.py",
        "summarize_layerwise_single_task_retention.py",
        "--checkpoint-round 4500",
        "summarize_layerwise_eedi_open_t3_t6_repair.py",
        "scientific_gate_pass",
    ):
        assert fragment in SOURCE


def test_watcher_disambiguates_equal_lr_curricula_by_job_path() -> None:
    assert '-path "*${job}*/eval/formal_screen/predictions.jsonl"' in SOURCE
    assert "balanced_lr5e7" in SOURCE
    assert "strong_lr5e7" in SOURCE


def test_global_supervisor_keeps_t6_repair_watcher_alive() -> None:
    assert "layerwise_eedi_open_t3_t6_repair" in SUPERVISOR
    assert "watch_layerwise_eedi_open_t3_t6_repair.sh" in SUPERVISOR
