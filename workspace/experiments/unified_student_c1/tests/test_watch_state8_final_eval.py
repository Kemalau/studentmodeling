from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_controller_is_parallel_recoverable_and_semantic() -> None:
    text = (ROOT / "scripts/watch_state8_final_eval.sh").read_text()
    for fragment in (
        "seq -w 0 15",
        "state8-final-c4-seed13",
        "validate_state8_final_eval_amlt.py",
        "terminal_nonpass_job_names",
        "preserve_passed",
        "count_active_a100_requests.py",
        "available_user_slots",
        'touch "$state_dir/passed.$job"',
        'for job in "${all_jobs[@]}"',
        "waiting for A100 slots for final-eval wave",
        "wave=$((wave+1))",
        "merge_state8_final_eval.py",
        "judge_formal_t3_predictions.py",
        "--judge-backend trapi",
        "validate_t3_semantic_result.py",
        "diagnosis_c4_semantic",
        "summarize_state8_diagnosis_c4.py",
        "state8_closed_diagnosis_c4_semantic_multiseed_v1",
        "--diagnosis-c4",
        "validate_state8_final_eval.py",
        "scientific_gate_pass",
        "scientific_gate_failed",
        "state8_clean_prompt_teacher_provenance_gate_v2",
        "legacy_answer_adjacent_contracts",
        "--final-fit-receipt",
        "has_forward_progress_job",
        "paused_job_names",
        "amlt resume",
        "sys.stdout.write(''.join(value+'\\n' for value in values))",
        "MAX_RECOVERY_WAVES:-32",
        "SAME_QUESTION_FINAL_EVAL",
        "state8-final-eedi-same-question-c4-final150",
        "validate_state8_eedi_same_question_final_eval_amlt.py",
        "eedi_same_question_c4_final150",
        "eedi_same_question_c4_semantic",
        "--eedi-same-question-c4",
            "use_studentmodeling_feeds_auth.sh",
        'fit_dir="${FIT_DIR:-',
        'state_dir="${STATE_DIR:-',
        'base_experiment="${BASE_EXPERIMENT:-',
        'recovery_experiment_prefix="${RECOVERY_EXPERIMENT_PREFIX:-',
        'recovery_experiment_suffix="${RECOVERY_EXPERIMENT_SUFFIX:-',
        'experiment="${recovery_experiment_prefix}${wave}${recovery_experiment_suffix}"',
    ):
        assert fragment in text
    assert "amlt cancel" not in text
    assert text.index("paused_job_names") < text.index("has_forward_progress_job")
    assert text.index('exec 8>"$project_dir/results/feeds_submission_global.lock"') < text.index(
        'wait_quota "${#names[@]}"'
    )


def test_same_question_wrapper_binds_finalfit_config_and_twentieth_job() -> None:
    text = (ROOT / "scripts/watch_state8_eedi_same_question_final_eval.sh").read_text()
    for fragment in (
        "state8_eedi_same_question_finalfit_world2_20260811",
        "feeds_state8_eedi_same_question_final_eval_20260811.yaml",
        "SAME_QUESTION_FINAL_EVAL=1",
        "watch_state8_final_eval.sh",
    ):
        assert fragment in text
