from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_state_kd_watcher_is_preflighted_recoverable_and_gate_bound() -> None:
    text = (ROOT / "scripts/watch_layerwise_state_kd.sh").read_text(encoding="utf-8")
    for fragment in (
        "wait_quota",
        "validate_layerwise_state_kd_amlt.py",
        "classify_experiment_status",
        "all_expected_jobs_pass",
        "wave=$((wave+1))",
        "judge_formal_t3_predictions.py",
        "--loss-profile state_kd",
        "scientific_gate_pass",
    ):
        assert fragment in text
