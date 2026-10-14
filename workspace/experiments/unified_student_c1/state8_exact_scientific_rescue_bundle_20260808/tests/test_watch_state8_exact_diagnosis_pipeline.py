from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_state8_exact_diagnosis_pipeline.sh").read_text()


def test_watcher_uses_independent_trapi_without_exact_shortcuts() -> None:
    for fragment in (
        "--judge-backend trapi",
        "llm_judge_api_or_cache",
        "llm_judge_exact_shortcuts",
        "shortcuts==0",
        "semantic_judge_authoritative",
        "exact_match_authoritative",
        "canonical_label_role",
        "summarize_state8_diagnosis_c4.py",
    ):
        assert fragment in SOURCE


def test_watcher_recovers_remote_failures_without_ambiguous_duplicate_submit() -> None:
    for fragment in (
        "max_recovery_waves=24",
        "submit_recovery()",
        "feeds_submission_global.lock",
        "validate_state8_exact_diagnosis_pipeline_amlt.py",
        'amlt run "$config" --dump -t Feeds',
        'amlt --json-tables ti sing -t Feeds',
        'amlt resume "$experiment" ":$job"',
        "Never issue an ambiguous duplicate",
        'printf \'%s\\n\' "$experiment" > "$state/active_experiment.txt"',
        "exact diagnosis pipeline exhausted recovery waves",
        'sleep_unlocked(){ sleep "$1" 9>&-; }',
    ):
        assert fragment in SOURCE
