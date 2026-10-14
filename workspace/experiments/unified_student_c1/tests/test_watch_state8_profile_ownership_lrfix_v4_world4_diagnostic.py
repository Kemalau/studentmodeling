from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/watch_state8_profile_ownership_lrfix_v4_world4_diagnostic.sh"


def test_world4_watcher_is_fail_closed_and_checks_exact_history_contract() -> None:
    text = WATCHER.read_text(encoding="utf-8")
    assert "state8-profile-ownership-lrfix-v4-world4-diagnostic-20260821" in text
    assert 'payload.get("distributed_world_size") == 4' in text
    assert 'payload.get("micro_steps") == 8' in text
    assert 'positions == list(range(8))' in text
    assert 'payload.get("rank_assignment_consistent") is True' in text
    assert 'payload.get("history_integrity_pass") is True' in text
    assert 'expected one distributed diagnostic artifact' in text
    assert 'marker = "complete" if not failures else "needs_attention"' in text
    assert '"$state/terminal_summary.json"' in text


def test_world4_watcher_never_submits_or_uses_device_code() -> None:
    text = WATCHER.read_text(encoding="utf-8")
    assert "amlt run" not in text
    assert "device-code" not in text
    assert "use-device-code" not in text
