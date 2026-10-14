from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_completed_scientific_failures_are_selector_inputs() -> None:
    source = (ROOT / "scripts/watch_state8_curriculum_selection.sh").read_text()
    assert 'if [[ -s "$path" ]]; then' in source
    assert "Scientific gate failures belong in the selector's failed_checks" in source
    assert (
        'fail "checkpoint screen needs attention before producing summary path=$parent"'
        in source
    )
    assert '[[ ! -f "$parent/needs_attention" ]] || fail' not in source
