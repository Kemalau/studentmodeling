from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_promotion_supervisor_is_gate_driven_and_confirmatory_closed() -> None:
    source = (
        ROOT / "scripts/promote_state8_profile_ownership_lrfix_v4_formal.sh"
    ).read_text()
    assert "validate_state8_profile_ownership_lrfix_v4_promotion.py" in source
    assert "validate_state8_profile_ownership_lrfix_v4_formal_amlt.py" in source
    assert '[[ ! -f "$smoke_state/complete"' in source
    assert '[[ ! -f "$world4_state/complete"' in source
    assert '"world_size_4": diagnostic.get("distributed_world_size") == 4' in source
    assert '"micro_steps_8": diagnostic.get("micro_steps") == 8' in source
    assert 'diagnostic.get("global_positions_seen") == list(range(8))' in source
    assert 'diagnostic.get("rank_assignment_consistent") is True' in source
    assert 'diagnostic.get("history_integrity_pass") is True' in source
    assert "amlt --json-tables ti sing -t Feeds -a" in source
    assert 'amlt run "$config" --dump -t Feeds' in source
    assert 'amlt run "$config" "$experiment" -t Feeds -y' in source
    assert "watch_state8_student_profile_readout.sh" in source
    assert "device-code" not in source
    assert "xes_state8_profile_confirmatory_v1" not in source


def test_formal_job_is_idempotent_and_submission_locked() -> None:
    source = (
        ROOT / "scripts/promote_state8_profile_ownership_lrfix_v4_formal.sh"
    ).read_text()
    assert 'status "$experiment"' in source
    assert "feeds_submission_global.lock" in source
    assert 'touch "$state/promoted"' in source
    assert '"$state/suppressed"' in source


def test_formal_promotion_selects_world4_config() -> None:
    source = (
        ROOT / "scripts/promote_state8_profile_ownership_lrfix_v4_formal.sh"
    ).read_text()
    assert "feeds_state8_profile_ownership_lrfix_v4_formal_world4_20260821.yaml" in source
