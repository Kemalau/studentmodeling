from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_state8_eedi_same_question_full_mse1.sh"


def test_full_watcher_is_a_fail_closed_mode_transform():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "watch_state8_eedi_same_question_causal_repair.sh" in text
    assert "state8-eedi-same-question-full-mse1-20260811" in text
    assert "state8-eedi-same-question-full-mse1-seed3071" in text
    assert 'assert count > 0' in text
    assert 'bash -n "$rendered"' in text
