from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/watch_state8_profile_ownership_lrfix_v4_world4_hashfix.sh"


def test_hashfix_watcher_preserves_exact_world4_gate_and_promotes_receipt() -> None:
    text = WATCHER.read_text(encoding="utf-8")
    assert "state8-profile-ownership-lrfix-v4-world4-hashfix-20260821" in text
    assert 'payload.get("distributed_world_size") == 4' in text
    assert 'payload.get("micro_steps") == 8' in text
    assert 'payload.get("global_positions_seen") == list(range(8))' in text
    assert 'payload.get("rank_assignment_consistent") is True' in text
    assert 'payload.get("history_integrity_pass") is True' in text
    assert 'terminal_summary.hash_drift_failure.json' in text
    assert 'touch "$original/complete" "$state/promoted_original_world4"' in text
    assert "promote_state8_profile_ownership_lrfix_v4_formal.sh" in text


def test_hashfix_watcher_is_monitor_only_and_never_uses_device_code() -> None:
    text = WATCHER.read_text(encoding="utf-8")
    assert "amlt run" not in text
    assert "device-code" not in text
    assert "use-device-code" not in text
