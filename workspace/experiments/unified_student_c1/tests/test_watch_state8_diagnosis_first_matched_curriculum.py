from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_state8_diagnosis_first_matched_curriculum.sh").read_text()


def test_matched_watcher_uses_only_semantic_authority() -> None:
    for fragment in (
        "curriculum_submitted",
        "judge_formal_t3_predictions.py",
        "--judge-backend trapi",
        "state8_diagnosis_first_matched_final_diagnosis_c4_corrected_20260810",
        "corrected_c4/c4_semantic/summary.json",
        "corrected_evaluator_sha256=660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca",
        "answer_c4_150/recurrent_causal_gate.json",
        "'answer_c4_pass'",
        "select_state8_diagnosis_first_matched_curriculum.py",
        "teacher_full_gate.json",
        "teacher_full_provenance_gate.json",
        "selection_rc",
        "state8_diagnosis_first_matched_v1/selection.json",
        "'semantic_judge_authoritative':True",
        "'exact_match_authoritative':False",
        "'final_inference_route':'strict_state_only'",
        "submit_recovery",
        "feeds_submission_global.lock",
        "max_recovery_waves=24",
        "active_experiment.txt",
        "monitor_layerwise_unified_training_progress.py",
        "STATE8_MATCHED_CURRICULUM_STALL_SECONDS:-2700",
        "STATE8_MATCHED_CURRICULUM_STARTUP_STALL_SECONDS:-7200",
        "STATE8_MATCHED_CURRICULUM_POSTTRAIN_STALL_SECONDS:-21600",
        'amlt logs view -F "$log_file"',
        'amlt cancel "$experiment" ":$job" -y',
        "cancelled stalled/unhealthy matched curriculum for exact checkpoint recovery",
    ):
        assert fragment in SOURCE
    assert "*/eval/diagnosis_c4_20/judge_predictions.jsonl" not in SOURCE
