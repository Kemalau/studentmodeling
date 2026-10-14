from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "scripts/watch_state8_balanced_teacher_curriculum.sh").read_text()


def test_balanced_controller_waits_for_teacher_recovers_and_verifies_variant() -> None:
    assert "teacher_gate_pass" in TEXT
    assert "state8-qualified-teacher-balanced-curriculum-seed42" in TEXT
    assert "MAX_RECOVERY_WAVES:-32" in TEXT
    assert "all_expected_jobs_pass" in TEXT
    assert "has_forward_progress_job" in TEXT
    assert "paused_job_names" in TEXT
    assert "amlt resume" in TEXT
    assert "wait_quota" in TEXT
    assert 'feeds_submission_global.lock' in TEXT
    assert TEXT.index('feeds_submission_global.lock') < TEXT.index(
        'amlt status --no-update-db "$experiment"'
    )
    assert "replay_variant']=='balanced'" in TEXT
    assert "state8_qualified_teacher_balanced_curriculum_immutable_v1" in TEXT
    assert '[[ -s "$state/active_experiment.txt" ]]' in TEXT
    assert 'recovery([0-9]+)-20260805' in TEXT
    assert "amlt cancel" not in TEXT
