from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watcher_recovers_and_runs_authoritative_full_judge() -> None:
    source = (ROOT / "scripts/watch_candidate_lr5e6_statepath_full_t3.sh").read_text()
    assert "terminal_nonpass" not in source
    assert "all_expected_jobs_pass" in source
    assert "recovery${wave}" in source
    assert "validate_candidate_lr5e6_statepath_full_t3_amlt.py" in source
    assert "feeds_submission_global.lock" in source
    assert "judge_formal_t3_predictions.py" in source
    assert "validate_t3_semantic_result.py" in source
    assert "full1494" in source
    assert "scientific_gate_pass" in source
    assert "scientific_gate_failed" in source
    assert "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525" in source
