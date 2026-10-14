from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_exact_rescue_watcher_is_conditional_resumable_and_authoritative() -> None:
    source = (ROOT / "scripts/watch_state8_exact_scientific_rescue.sh").read_text()
    for fragment in (
        "selection_passes",
        "selection_fails",
        'touch "$state/not_required" "$state/complete"',
        "materialize_state8_exact_scientific_rescue_amlt.py",
        "validate_state8_exact_scientific_rescue_amlt.py",
        "feeds_submission_global.lock",
        "count_active_a100_requests.py",
        "available_user_slots",
        "submitted_jobs.txt",
        'amlt run "$config" "${selectors[@]}" --dump -t Feeds',
        'amlt --json-tables ti sing -t Feeds',
        "preserve_passed",
        "max_recovery_waves=24",
        "--judge-backend trapi",
        "summarize_state8_diagnosis_c4.py",
        "select_state8_exact_scientific_rescue.py",
        "state8_exact_scientific_rescue_v1/selection.json",
        'sleep_unlocked(){ sleep "$1" 9>&-; }',
        ".azure-studentmodeling-feeds",
    ):
        assert fragment in source
