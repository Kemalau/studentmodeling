from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watcher_is_fail_closed_and_semantic() -> None:
    text = (ROOT / "scripts/watch_state8_semantic_causal_screen.sh").read_text()
    assert "state8-semantic-causal-strongbase-seed45-20260806" in text
    assert "has_forward_progress_job" in text
    assert "all_expected_jobs_pass" in text
    assert "--judge-backend trapi" in text
    assert "summarize_state8_diagnosis_c4.py" in text
    assert "semantic causal job reached a non-pass terminal state" in text
