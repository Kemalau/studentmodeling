from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_watcher_recovers_and_runs_semantic_judges() -> None:
    text = (ROOT / "scripts/watch_state8_semantic_candidate.sh").read_text()
    for fragment in (
        "EXPERIMENT is required",
        "CONFIG_PATH is required",
        'validator="${VALIDATOR:-scripts/validate_state8_semantic_causal_screen_amlt.py}"',
        "submit_recovery",
        '"$python_bin" "$validator"',
        "all_expected_jobs_pass",
        "--judge-backend trapi",
        "summarize_state8_diagnosis_c4.py",
        "publish_receipt.json",
        'state="${STATE_DIR:-',
    ):
        assert fragment in text
