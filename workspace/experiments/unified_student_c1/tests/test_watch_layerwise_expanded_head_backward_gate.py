from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts/watch_layerwise_expanded_head_backward_gate.sh").read_text()


def test_three_real_rows_are_required_before_release() -> None:
    required = (
        "all_expected_jobs_pass(sys.argv[1], 3)",
        "expanded_head_backward_receipt.json",
        "549: (177752, ':233:1')",
        "653: (110499, ':0002079:malrule_identity')",
        "721: (68865, ':0002130:malrule_identity')",
        "value['runtime_sha256'] == runtime",
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        'rm -f "$final_fit/needs_attention" "$rehearsal/needs_attention"',
    )
    assert all(value in SCRIPT for value in required)


def test_failure_does_not_blindly_resubmit() -> None:
    assert "amlt run" not in SCRIPT
    assert "amlt cancel" not in SCRIPT
    assert "failure_logs" in SCRIPT


def test_success_releases_both_training_controllers() -> None:
    assert "watch_layerwise_task_retention_rehearsal.sh" in SCRIPT
    assert "watch_layerwise_unified_final_fit.sh" in SCRIPT
    assert "MAX_RESILIENT_RECOVERIES=12" in SCRIPT
