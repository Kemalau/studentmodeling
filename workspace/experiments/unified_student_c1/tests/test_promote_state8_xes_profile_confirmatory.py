from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confirmatory_promotion_is_fail_closed_and_one_shot() -> None:
    source = (ROOT / "scripts/promote_state8_xes_profile_confirmatory.sh").read_text()
    assert "reusable_interface_scientific_gate_pass" in source
    assert "all_preregistered_gates" in source
    assert "XES test remains untouched" in source
    assert "feeds_submission_global.lock" in source
    assert "amlt run" in source
    assert "-y" in source


def test_confirmatory_watcher_never_resubmits() -> None:
    source = (ROOT / "scripts/watch_state8_xes_profile_confirmatory.sh").read_text()
    assert "amlt run" not in source
    assert "test_labels_used_for_selection" in source
    assert "payload.get('students')=={'train':800,'valid':200,'test':466}" in source
