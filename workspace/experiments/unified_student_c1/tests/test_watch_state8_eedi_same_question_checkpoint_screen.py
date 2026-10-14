from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_state8_eedi_same_question_checkpoint_screen.sh"


def test_watcher_defines_four_independent_screens_and_semantic_judging():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "SCREEN_ARM" in text and "CHECKPOINT_STEP" in text
    assert "updater" in text and "full" in text
    assert "malrule_diagnosis_c4_20" in text
    assert "eedi_same_question_c4_25" in text
    assert "--judge-backend trapi" in text
    assert "max_recovery=3" in text
    assert 'while [[ ! -f "$state/snapshot_complete" ]]' in text
    assert "preserving checkpoint and retrying" in text
    assert "s<={'','unknown'}: print('unknown')" in text
    assert "preserving remote screen and retrying" in text
