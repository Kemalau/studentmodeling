from pathlib import Path

from scripts.state8_eedi_curriculum_liveness import should_cancel


ROOT = Path(__file__).resolve().parents[1]


def test_liveness_guard_requires_backend_running_and_scientific_progress_evidence():
    source = (
        ROOT / "scripts/watch_state8_eedi_curriculum_liveness.sh"
    ).read_text(encoding="utf-8")
    for fragment in (
        "states=={'running'}",
        "monitor_layerwise_unified_training_progress.py",
        "state8_eedi_curriculum_liveness.py",
        "STATE8_EEDI_LIVENESS_STALL_SECONDS:-2700",
        "STATE8_EEDI_LIVENESS_STARTUP_STALL_SECONDS:-7200",
        "STATE8_EEDI_LIVENESS_POSTTRAIN_STALL_SECONDS:-21600",
        'amlt cancel "$experiment" ":$job" -y',
        "exact-resume owner will recover",
    ):
        assert fragment in source
    assert "amlt run" not in source
    assert "--force-status-update" in source
    assert "logs view -F" in source
    assert "health.get('healthy') is False" not in source
    assert "causal_path_observed_when_round1_present" not in source


def test_liveness_guard_covers_both_curricula_and_is_single_owner_safe():
    source = (
        ROOT / "scripts/watch_state8_eedi_curriculum_liveness.sh"
    ).read_text(encoding="utf-8")
    assert "state8_diagnosis_first_eedi_curriculum_20260812" in source
    assert "state8_diagnosis_first_eedi_current_query_curriculum_20260812" in source
    assert "flock -n 9" in source
    assert "cancel_requested" in source
    assert "use_studentmodeling_feeds_auth.sh" in source


def progress(**overrides):
    value = {
        "round": 3920,
        "total_rounds": 5000,
        "progress_fraction": 0.784,
        "seconds_since_progress": 0.0,
        "health": {
            "healthy": False,
            "checks": {
                "finite_losses": True,
                "no_runtime_error_signature": True,
                "memory_below_a100_80gb": True,
                "causal_path_observed_when_round1_present": False,
            },
        },
    }
    value.update(overrides)
    return value


def decision(value):
    return should_cancel(
        value,
        stall_seconds=2700,
        startup_stall_seconds=7200,
        posttrain_stall_seconds=21600,
    )


def test_deferred_causal_schedule_advisory_does_not_cancel_advancing_training():
    assert decision(progress()) is False


def test_universal_fatal_signals_and_real_stall_cancel():
    oom = progress()
    oom["health"]["checks"]["memory_below_a100_80gb"] = False
    assert decision(oom) is True
    assert decision(progress(seconds_since_progress=2700.0)) is True


def test_phase_specific_thresholds_are_preserved():
    assert decision(
        progress(phase="awaiting_first_training_record", seconds_since_progress=7199)
    ) is False
    assert decision(
        progress(phase="awaiting_first_training_record", seconds_since_progress=7200)
    ) is True
    assert decision(progress(round=5000, progress_fraction=1.0, seconds_since_progress=21599)) is False
    assert decision(progress(round=5000, progress_fraction=1.0, seconds_since_progress=21600)) is True
