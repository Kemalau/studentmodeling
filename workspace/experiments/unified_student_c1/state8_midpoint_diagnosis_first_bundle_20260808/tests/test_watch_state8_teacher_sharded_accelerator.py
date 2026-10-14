from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sharded_accelerator_is_fail_closed_and_unlocks_primary_pipeline() -> None:
    source = (ROOT / "scripts/watch_state8_teacher_sharded_accelerator.sh").read_text()
    assert "merge_state8_teacher_eval_shards.py" in source
    assert "--judge-backend trapi" in source
    assert "validate_state8_teacher_gate.py" in source
    assert "teacher_gate_publish.lock" in source
    assert "state8-qualified-teacher-curriculum-20260805" in source
    assert "terminal_nonpass_job_names" in source
    assert 'touch "$teacher_state/stage_b_submitted" "$teacher_state/complete"' in source
