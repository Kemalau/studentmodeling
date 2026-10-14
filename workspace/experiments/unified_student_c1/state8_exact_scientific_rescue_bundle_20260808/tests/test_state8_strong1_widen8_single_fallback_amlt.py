from pathlib import Path

from scripts.validate_state8_strong1_widen8_single_fallback_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_single_fallback_is_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_strong1_widen8_single_fallback_20260808.yaml")


def test_single_fallback_uses_backfill_friendly_resumable_slices() -> None:
    source = (
        ROOT / "amlt/feeds_state8_strong1_widen8_single_fallback_20260808.yaml"
    ).read_text()
    assert "max_run_duration_seconds: 14400" in source
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "select_latest_valid_single_checkpoint.py" in launcher
    assert '--save-steps "$STATE8_SAVE_STEPS"' in launcher
    assert '--resume-from-checkpoint "$resume_checkpoint"' in launcher
    assert "STATE8_SAVE_STEPS % 4" in launcher
    assert "STATE8_SAVE_STEPS: '200'" in source


def test_single_fallback_monitor_runs_authoritative_followup() -> None:
    source = (ROOT / "scripts/watch_state8_strong1_widen8_single_fallback.sh").read_text()
    for fragment in (
        "validate_state8_strong1_widen8_single_eval_amlt.py",
        "judge_formal_t3_predictions.py",
        "summarize_state8_diagnosis_c4.py",
        "select_state8_strong1_widen8_bridge.py",
        "single_lr5e7_seed60",
        'candidate_root="$state/$candidate_slug"',
        "scientific_gate_pass",
        "scientific_gate_failed",
    ):
        assert fragment in source


def test_single_fallback_progress_uses_actual_amlt_attempt_duration() -> None:
    source = (ROOT / "scripts/watch_state8_strong1_widen8_single_fallback.sh").read_text()
    assert "attempt_max_run_seconds" in source
    assert "max_run_duration_seconds" in source
    assert '--max-run-seconds "$attempt_max_run_seconds"' in source
    assert "--max-run-seconds 345600" not in source
