from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "scripts/watch_state8_qualified_promotion_selection.sh").read_text()


def test_selector_short_circuits_by_preregistered_priority_and_fails_closed() -> None:
    assert "state8_qualified_promotion_eval_20260805" in TEXT
    assert "state8_qualified_balanced_promotion_eval_20260805" in TEXT
    assert "state8_qualified_highkd_promotion_eval_20260805" in TEXT
    assert "--highkd" in TEXT
    assert "select_state8_qualified_promotion.py" in TEXT
    assert "scientific_gate_pass" in TEXT
    assert "scientific_gate_failed" in TEXT
    assert "needs_attention" in TEXT
    assert 'wait_upstream "$primary"' in TEXT
    assert 'if ! gate_passes "$primary/promotion_gate.json"' in TEXT
    assert 'wait_upstream "$highkd"' in TEXT
    assert 'if ! gate_passes "$highkd/promotion_gate.json"' in TEXT
    assert 'wait_upstream "$balanced"' in TEXT
    assert "Unselected arms keep running" in TEXT
    assert TEXT.index('wait_upstream "$primary"') < TEXT.index('wait_upstream "$highkd"')
    assert TEXT.index('wait_upstream "$highkd"') < TEXT.index('wait_upstream "$balanced"')
