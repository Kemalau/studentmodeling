from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "scripts/watch_state8_highkd_teacher_curriculum.sh").read_text()


def test_highkd_watcher_recovers_and_validates_scientific_identity() -> None:
    assert "MAX_RECOVERY_WAVES:-32" in TEXT
    assert "state8-qualified-teacher-highkd-curriculum-recovery" in TEXT
    assert "validate_state8_qualified_teacher_curriculum_amlt.py" in TEXT
    assert "online_kd_weight']==1.0" in TEXT
    assert "state8_qualified_teacher_curriculum_highkd_immutable_v1" in TEXT
    assert "raw_history_visible_to_decoder'] is False" in TEXT
    assert "needs_attention" in TEXT
    assert "all_expected_jobs_pass" in TEXT
    assert "has_forward_progress_job" in TEXT
    assert 'submit_wave "$experiment" "$wave"' in TEXT
    assert "initial/reconnect high-KD submission failed" in TEXT


def test_highkd_watcher_submits_before_monitoring_status() -> None:
    """A missing base experiment must not be polled forever."""
    reconcile = TEXT.index('submit_wave "$experiment" "$wave"')
    monitor = TEXT.index("while true; do", reconcile)
    status = TEXT.index('status="$(timeout 90 amlt status "$experiment"', monitor)
    assert reconcile < monitor < status
