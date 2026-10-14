from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_early_selector_waits_for_complete_four_cell_matrix():
    text=(ROOT/"scripts/watch_state8_eedi_same_question_early_selection.sh").read_text()
    assert "for arm in updater full" in text
    assert "for step in 20 100" in text
    assert "select_state8_eedi_same_question_checkpoint.py" in text
    assert "no_viable_early_checkpoint" in text
    assert "plan_state8_eedi_same_question_repair.py" in text
    assert "repair_required" in text
