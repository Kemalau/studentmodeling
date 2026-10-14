from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rescue_watcher_preserves_outputs_and_uses_authoritative_semantic_judges() -> None:
    source = (ROOT / "scripts/watch_state8_matched_checkpoint2000_rescue.sh").read_text()
    for fragment in (
        "state8-matched-cp2000-causal-rescue-fix2-20260810",
        "state8-matched-cp2000-rescue-semantic-preserve",
        "state8-matched-cp2000-rescue-causal-taskanchor",
        "amlt results download",
        "--judge-backend trapi",
        "summarize_state8_diagnosis_c4.py",
        "semantic_judge_authoritative",
        "exact_match_authoritative",
        "canonical_label_role",
        "answer_c4_50/recurrent_causal_gate.json",
        "selected_for_followup",
        "answer_diagnostic_gate",
        "diagnosis_diagnostic_gate",
        "requires 150-pair confirmation before promotion",
        'sleep_unlocked(){ sleep "$1" 9>&-; }',
        "submit_jobs",
        "feeds_submission_global.lock",
        "max_recovery_waves=24",
        "active_experiment.txt",
        "transient rescue recovery submission failure",
        "submitted rescue recovery wave=",
    ):
        assert fragment in source
