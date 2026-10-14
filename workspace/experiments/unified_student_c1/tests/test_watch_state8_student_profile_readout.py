from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watcher_allows_an_explicit_browser_auth_cache() -> None:
    source = (ROOT / "scripts/watch_state8_student_profile_readout.sh").read_text()
    assert (
        "auth=${PROFILE_READOUT_AUTH:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}"
        in source
    )
    assert 'test -d "$auth"' in source


def test_watcher_downloads_and_independently_audits_terminal_artifacts() -> None:
    source = (ROOT / "scripts/watch_state8_student_profile_readout.sh").read_text()
    assert "amlt results download" in source
    assert "audit_state8_future_profile_v2.py" in source
    assert "independent_audit.json" in source
    assert 'rm -f "$state/complete" "$state/needs_attention"' in source
    assert "opposite_marker" in source
    assert "distributed_diagnostic_completion.json" in source
    assert "diagnostic_complete=passed" in source


def test_unknown_amlt_status_is_not_treated_as_terminal() -> None:
    source = (ROOT / "scripts/watch_state8_student_profile_readout.sh").read_text()
    assert "success={'pass','passed'}" in source
    assert "failure={'failed','fail','killed','cancelled','canceled','stopped'}" in source
    assert "all(state in success | failure for state in states)" in source
