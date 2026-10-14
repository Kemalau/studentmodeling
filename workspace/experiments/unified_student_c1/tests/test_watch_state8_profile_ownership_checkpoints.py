from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_profile_ownership_checkpoint_watcher_uses_receipt_last_snapshots() -> None:
    source = (ROOT / "scripts/watch_state8_profile_ownership_checkpoints.sh").read_text()
    assert "snapshot_state8_checkpoint_blob.py" in source
    assert "checkpoint_completion.json" in source
    assert "checkpoint_snapshots/state8_future_profile_ownership_seed8801_v3" in source
    assert "step >= 100 && step <= 1000 && step % 100 == 0" in source
    assert "--auth-mode login" in source
    assert "device" not in source.casefold()


def test_profile_ownership_checkpoint_watcher_never_mutates_source() -> None:
    source = (ROOT / "scripts/watch_state8_profile_ownership_checkpoints.sh").read_text()
    forbidden = ("blob delete", "rm -rf \"$source_root", "azcopy remove")
    assert all(fragment not in source for fragment in forbidden)
