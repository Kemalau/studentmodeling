from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_milestone_watcher_is_conditional_parallel_resumable_and_semantic() -> None:
    source = (ROOT / "scripts/watch_state8_exact_consolidation_milestones.sh").read_text()
    for fragment in (
        'touch "$state/not_required" "$state/complete"',
        "validate_state8_exact_consolidation_milestones_amlt.py",
        "feeds_submission_global.lock",
        "count_active_a100_requests.py",
        "available_user_slots",
        "reserve=3",
        "state8_exact_scientific_rescue_20260808",
        "submitted_jobs.txt",
        'amlt run "$config" "${selectors[@]}" --dump -t Feeds',
        'amlt --json-tables ti sing -t Feeds',
        "preserve_passed",
        "max_recovery_waves=24",
        "--judge-backend trapi",
        "summarize_state8_diagnosis_c4.py",
        "select_state8_exact_consolidation_milestones.py",
        'sleep_unlocked(){ sleep "$1" 9>&-; }',
        ".azure-studentmodeling-feeds",
    ):
        assert fragment in source
