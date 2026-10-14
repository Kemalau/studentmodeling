from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gate_accepts_pinned_legacy_receipt_but_checks_new_fields_when_present():
    source = (
        ROOT / "scripts/supervise_state8_ownership_world4_diagnostic2.sh"
    ).read_text(encoding="utf-8")
    assert "value['history_rows']==8" in source
    assert "value['history_unique_global_positions']==8" in source
    assert "value['history_global_positions_contiguous'] is True" in source
    assert "value['history_global_position_start']==0" in source
    assert "value['history_global_position_end']==7" in source
    assert "if 'distributed_world_size' in value:" in source
    assert "if 'rank_assignment_consistent' in value:" in source
    assert "if 'global_positions_seen' in value:" in source
