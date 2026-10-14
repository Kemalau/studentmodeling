from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cp5000_watcher_uses_authoritative_semantic_judge() -> None:
    text = (ROOT / "scripts/watch_state8_cp5000_semantic_causal_repair.sh").read_text()
    assert "state8-cp5000-semantic-causal-repair-20260812" in text
    assert "--judge-backend trapi" in text
    assert "--trapi-deployment gpt-4o_2024-11-20" in text
    assert "summarize_state8_diagnosis_c4.py" in text
    assert "exact_match_authoritative':False" in text
    assert "select_state8_eedi_same_question_candidate.py" in text


def test_cp5000_watcher_has_no_exact_match_gate() -> None:
    text = (ROOT / "scripts/watch_state8_cp5000_semantic_causal_repair.sh").read_text()
    assert "exact_match_authoritative':True" not in text
    assert "semantic_judge_authoritative':True" in text
