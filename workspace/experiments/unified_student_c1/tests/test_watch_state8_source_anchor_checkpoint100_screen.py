from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint100_watcher_is_hash_bound_and_validates_recovery() -> None:
    text = (ROOT / "scripts/watch_state8_source_anchor_checkpoint100_screen.sh").read_text()
    assert "186be7cde8a544e21a227e6ef51df160ef8be4e9c31f053dc93f8d435732029b" in text
    assert "state8-source-anchor-cp100" in text
    assert 'replace("checkpoint20", "checkpoint100", 1)' in text
    assert "validate_state8_source_anchor_checkpoint100_screen_amlt.py" in text
    assert 'bash "$rendered"' in text
