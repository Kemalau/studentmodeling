from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_remote_launcher_uses_ownership_trainer_and_fails_closed() -> None:
    launcher = (ROOT / "scripts/run_state8_student_profile_readout.sh").read_text()
    assert "python -u scripts/train_state8_profile_ownership_repair.py" in launcher
    assert "test -f \"$RUN_DIR/profile_ownership_repair.json\"" in launcher
    assert "o['applied_rows']>0" in launcher


def test_remote_launcher_has_v5_specific_fail_closed_contract() -> None:
    launcher = (ROOT / "scripts/run_state8_student_profile_readout.sh").read_text()
    assert 'profile_ownership_v5.json' in launcher
    assert 'cp "$RUN_DIR"/*.json "$AMLT_OUTPUT_DIR/"' in launcher
    assert "v['profile_numeric_head_frozen'] is True" in launcher
    assert "v['multi_negative_profile_rows']>0" in launcher
    assert "p['ownership_geometry_exposures']==0" in launcher


def test_remote_launcher_has_bounded_distributed_diagnostic_mode() -> None:
    launcher = (ROOT / "scripts/run_state8_student_profile_readout.sh").read_text()
    assert "PROFILE_READOUT_DIAGNOSTIC_STOP_ROUND" in launcher
    assert "--diagnostic-stop-after-round" in launcher
    assert "distributed_diagnostic_completion.json" in launcher


def test_remote_launcher_has_bounded_single_gpu_ownership_diagnostic() -> None:
    launcher = (ROOT / "scripts/run_state8_student_profile_readout.sh").read_text()
    assert "PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS" in launcher
    assert "--diagnostic-max-micro-steps" in launcher
    assert "scripts/early_audit_state8_profile_ownership_checkpoint.py" in launcher
    assert "checkpoint_early_audit/metrics.json" in launcher


def test_v3_config_pins_the_corrected_launcher_snapshot() -> None:
    import hashlib

    launcher = ROOT / "scripts/run_state8_student_profile_readout.sh"
    digest = hashlib.sha256(launcher.read_bytes()).hexdigest()
    config = (ROOT / "amlt/feeds_state8_future_profile_ownership_v3_20260820.yaml").read_text()
    assert digest in config
