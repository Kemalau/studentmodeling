from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "scripts/watch_state8_primary_teacher_curriculum.sh").read_text()


def test_primary_controller_is_gate_driven_recoverable_and_atomic() -> None:
    for fragment in (
        "teacher_gate_pass",
        "teacher_provenance_gate.json",
        "state8-qualified-teacher-curriculum-seed42",
        "MAX_RECOVERY_WAVES:-32",
        "feeds_submission_global.lock",
        "wait_quota",
        "all_expected_jobs_pass",
        "has_forward_progress_job",
        "paused_job_names",
        "amlt resume",
        "stage_b_finalize.lock",
        "replay_variant']=='t3_heavy'",
        "state8_qualified_teacher_curriculum_immutable_v1",
        "raw_history_visible_to_decoder",
        '[[ -s "$state/active_experiment.txt" ]]',
        'recovery([0-9]+)-20260805',
    ):
        assert fragment in TEXT
    assert TEXT.index("feeds_submission_global.lock") < TEXT.index(
        'amlt status --no-update-db "$experiment"'
    )
    assert "amlt cancel" not in TEXT
