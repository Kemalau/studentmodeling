from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_monitor_downloads_and_semantically_judges_both_panels() -> None:
    source = (ROOT / "scripts/watch_state8_diagnosis_causal_rescue.sh").read_text()
    for fragment in (
        "state8-diagnosis-causal-rescue-seed43-20260806",
        "has_forward_progress_job",
        "all_expected_jobs_pass",
        "amlt results download",
        "judge_formal_t3_predictions.py",
        "--judge-backend trapi",
        "summarize_state8_diagnosis_c4.py",
        "formal_semantic",
        "diagnosis_semantic",
        "MAX_RECOVERY_WAVES:-32",
        "validate_state8_diagnosis_causal_rescue_amlt.py",
        "classify_experiment_status",
        "wave=$((wave+1))",
    ):
        assert fragment in source
