from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/watch_external_state_teacher_eedi_full.sh"
SUPERVISOR = ROOT / "scripts/supervise_recurrent_unified_pipeline.sh"


def test_watcher_preserves_passed_siblings_and_recovers_only_failed_arm():
    text = WATCHER.read_text(encoding="utf-8")
    assert "passed_receipts" in text
    assert 'amlt results download "$experiment" --status pass' in text
    assert 'amlt run "$config" ":$job" --dump -t Feeds' in text
    assert 'submit_recovery "$job" "${waves[$job]}"' in text
    assert "all_expected_jobs_pass" in text
    assert "terminal_nonpass_job_names" in text
    assert "exhausted six recovery waves" in text


def test_watcher_runs_authoritative_full_t3_and_joint_selection():
    text = WATCHER.read_text(encoding="utf-8")
    assert "judge_formal_t3_predictions.py" in text
    assert "--judge-backend trapi" in text
    assert "summarize_external_state_teacher_diagnostic.py" in text
    assert "scientific_gate_pass" in text
    assert "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525" in text


def test_supervisor_owns_the_new_diagnostic():
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "layerwise_external_state_teacher_eedi_full" in text
    assert "watch_external_state_teacher_eedi_full.sh" in text
    assert "layerwise_gold_semantic_teacher_eedi_full" in text
    assert "feeds_gold_semantic_teacher_eedi_full_diagnostic_20260804.yaml" in text
