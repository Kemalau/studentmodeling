from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_controller_is_completion_and_capacity_gated() -> None:
    source = (ROOT / "scripts/watch_taskonly_qwen_resumable_recovery.sh").read_text()
    for fragment in (
        'completion_exists && return 2',
        'available >= 1',
        'flock -x 8',
        '--jobs "$recovery_job"',
        'checkpoint every 500 steps',
        'validate_completion',
        'expected_steps=27632',
        'expected_steps=13893',
        'expected_steps=19650',
    ):
        assert fragment in source


def test_recovery_controller_never_cancels_healthy_original_jobs() -> None:
    source = (ROOT / "scripts/watch_taskonly_qwen_resumable_recovery.sh").read_text()
    assert "amlt cancel" not in source
    assert "original job completed; recovery not submitted" in source


def test_baseline_supervisor_restarts_all_three_recovery_controllers() -> None:
    source = (ROOT / "scripts/supervise_taskonly_baselines.sh").read_text()
    assert "ensure_resumable_recovery" in source
    assert "for task in T1 T2 T4; do ensure_resumable_recovery" in source
