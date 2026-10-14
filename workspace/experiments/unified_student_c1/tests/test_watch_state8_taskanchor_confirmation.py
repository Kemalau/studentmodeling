from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_watcher_is_resumable_and_semantic_authoritative() -> None:
    source = (ROOT / "scripts/watch_state8_taskanchor_confirmation.sh").read_text()
    for fragment in (
        "selected_for_full_confirmation",
        "feeds_submission_global.lock",
        "count_active_a100_requests.py",
        "preserve_passed",
        "state8-taskanchor-full-confirmation-recovery",
        "merge_state8_final_eval.py",
        "judge_formal_t3_predictions.py",
        "validate_t3_semantic_result.py",
        "summarize_state8_diagnosis_c4.py",
        "validate_state8_taskanchor_confirmation.py",
        "state8_taskanchor_confirmation_v1/selection.json",
    ):
        assert fragment in source
    assert "--judge-backend trapi" in source
    assert "exact" not in source.lower() or "exact_match_authoritative" not in source

