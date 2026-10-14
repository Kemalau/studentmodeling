from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_checkpoint200_watcher_is_diagnostic_only() -> None:
    text=(ROOT/'scripts/watch_state8_diagnosis_first_teacher_checkpoint200_screen.sh').read_text()
    for fragment in (
        'state8-diagnosis-first-teacher-checkpoint200-screen-20260810',
        'all_expected_jobs_pass', 'has_forward_progress_job', 'amlt resume',
        'judge_formal_t3_predictions.py', '--judge-backend trapi',
        'validate_state8_diagnosis_first_teacher_screen.py',
        'final Teacher remains authoritative',
    ):
        assert fragment in text
    assert 'amlt run' not in text
