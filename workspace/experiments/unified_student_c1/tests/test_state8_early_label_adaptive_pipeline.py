from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_early_label_milestones_are_receipt_last_and_non_submitting() -> None:
    text = (ROOT / "scripts/watch_state8_early_label_binding_milestones.sh").read_text()
    assert "milestones=(20 100 240 500)" in text
    assert "8301:early_label_low" in text and "8302:early_label_high" in text
    assert "snapshot_state8_checkpoint_blob.py" in text
    assert "checkpoint_completion.json" in text
    assert "amlt run" not in text


def test_early_label_selection_is_adaptive_and_fail_closed() -> None:
    text = (ROOT / "scripts/watch_state8_early_label_checkpoint_selection.sh").read_text()
    assert "for step in 20 100 240 500" in text
    assert "select_state8_early_label_checkpoint.py" in text
    assert "full early-label selection missing" in text
    assert 'touch "$state/scientific_gate_failed"' in text
    assert "watch_state8_curriculum_repair_checkpoint_screen.sh" in text
