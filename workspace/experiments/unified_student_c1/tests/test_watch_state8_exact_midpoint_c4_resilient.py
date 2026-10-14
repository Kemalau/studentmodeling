from pathlib import Path


def test_resilient_controller_preserves_recovers_merges_and_judges():
    source = Path("scripts/watch_state8_exact_midpoint_c4_resilient.sh").read_text()
    assert "preserve_passed" in source
    assert 'amlt results download "$experiment" ":$job"' in source
    assert "missing=(" in source
    assert "merge_state8_diagnosis_c4_shards.py" in source
    assert "judge_formal_t3_predictions.py" in source
    assert "summarize_state8_diagnosis_c4.py" in source
    assert "artifact_sha256" in source
    assert "state8_exact_diagnosis_midpoint_step2000_resilient_screen_v1" in source


def test_resilient_controller_uses_singleton_and_global_submission_locks():
    source = Path("scripts/watch_state8_exact_midpoint_c4_resilient.sh").read_text()
    assert 'flock -n 9' in source
    assert 'feeds_submission_global.lock' in source
    assert 'flock -x 8' in source
