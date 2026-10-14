from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts/watch_layerwise_recompute_mask_release.sh").read_text(
    encoding="utf-8"
)


def test_release_is_fail_closed_on_real_backward_receipt() -> None:
    required = (
        "all_expected_jobs_pass(sys.argv[1], 1)",
        "oom_recovery_receipt.json",
        "trainer_history.json",
        "endswith(':233:1')",
        "receipt['direct_tokens'] == 549",
        "receipt['direct_offload_used'] is True",
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        "rm -f \"$final_fit/needs_attention\"",
        "watch_layerwise_unified_final_fit.sh",
        "already-submitted validated recovery9",
    )
    assert all(value in SCRIPT for value in required)


def test_release_does_not_submit_training_directly() -> None:
    assert "amlt run" not in SCRIPT
    assert "amlt cancel" not in SCRIPT
