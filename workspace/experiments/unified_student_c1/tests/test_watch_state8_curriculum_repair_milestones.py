from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repair_milestone_watcher_covers_both_sources_and_receipt_last_snapshot():
    source = (ROOT / "scripts/watch_state8_curriculum_repair_milestones.sh").read_text()
    for seed in (5201, 5202, 5203, 6201, 6202, 6203):
        assert str(seed) in source
    assert "milestones=(20 100 240 500)" in source
    assert "Step 250" in source
    assert "impossible blob" in source
    assert "snapshot_state8_checkpoint_blob.py" in source
    assert "checkpoint_completion.json" in source
    assert "--auth-mode login" in source
    assert "amlt run" not in source
    assert "raw_history_visible_to_decoder" not in source
