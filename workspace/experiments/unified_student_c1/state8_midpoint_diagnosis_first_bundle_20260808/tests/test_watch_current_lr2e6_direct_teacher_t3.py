from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_direct_teacher_watcher_recovers_and_never_promotes() -> None:
    source = (ROOT / "scripts/watch_current_lr2e6_direct_teacher_t3.sh").read_text()
    assert "all_expected_jobs_pass" in source
    assert "validate_current_lr2e6_direct_teacher_t3_amlt.py" in source
    assert "judge_formal_t3_predictions.py" in source
    assert "eedi_semantic_accuracy" in source
    assert "eligible_for_final_gate':False" in source
    assert "direct-teacher T3 mechanism diagnostic complete" in source
