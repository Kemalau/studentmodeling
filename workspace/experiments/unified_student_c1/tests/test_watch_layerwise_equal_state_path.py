from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_layerwise_equal_state_path.sh").read_text()


def test_monitor_recovers_and_completes_independent_scientific_judgment() -> None:
    for fragment in (
        "validate_layerwise_equal_state_path_amlt.py",
        "recurrent_submit_guard",
        "amlt results download",
        "eval/c4/recurrent_causal_gate.json",
        "eval/t6_full/metrics.json",
        "judge_formal_t3_predictions.py",
        "summarize_layerwise_single_task_retention.py",
    ):
        assert fragment in SOURCE


def test_monitor_has_bounded_recovery_and_heartbeats() -> None:
    assert "heartbeat" in SOURCE
    assert "wave < 4" in SOURCE
    assert "needs_attention" in SOURCE
    assert 'active_experiment.txt' in SOURCE


def test_recovery_is_quota_gated_idempotent_and_survives_controller_restart() -> None:
    assert "wait_quota" in SOURCE
    assert "select_a100_80gb" in SOURCE
    assert "classify_experiment_status" in SOURCE
    assert "preexisting_not_resubmitted" in SOURCE
    assert '[[ ! -s "$state_dir/active_experiment.txt" ]]' in SOURCE
    assert 'printf \'%s\\n\' "$experiment" > "$state_dir/active_experiment.txt"' in SOURCE
    assert 'printf \'%s\\n\' "$base_experiment" > "$state_dir/active_experiment.txt"' not in SOURCE


def test_missing_initial_experiment_submits_wave_zero_instead_of_sleeping_forever() -> None:
    assert 'if [[ "$class" == missing ]]' in SOURCE
    assert '[[ "$experiment" == "$base_experiment" && "$wave" -eq 0 ]]' in SOURCE
    assert 'submit_recovery 0 || fail "initial state-path submission failed wave=0"' in SOURCE
    assert 'if [[ "$class" == ambiguous ]]' in SOURCE
    assert 'fail "ambiguous state-path experiment status' in SOURCE
