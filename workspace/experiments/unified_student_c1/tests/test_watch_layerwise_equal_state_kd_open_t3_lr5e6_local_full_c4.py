from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/watch_layerwise_equal_state_kd_open_t3_lr5e6_local_full_c4.sh"
).read_text()


def test_full_c4_pins_passed_state_kd_artifact_and_frozen_manifest() -> None:
    for fragment in (
        "equal_state_kd_open_t3_lr_scan/local_lr5e6_round3250_c4",
        "equal_state_kd_open_t3_lr_scan/local_lr5e6_round3250_c4_full",
        "checkpoint-round-3250",
        "c4_stage_a_gate_pairs_seed42_n150_v1.json",
        "--num-pairs 150",
        "--state-architecture layerwise",
        "--state-token-alias '<|vision_pad|>'",
        "local_gpu_readonly_preflight.lock",
        "equal-open-t3-state-kd-lr5e6",
    ):
        assert fragment in SCRIPT
