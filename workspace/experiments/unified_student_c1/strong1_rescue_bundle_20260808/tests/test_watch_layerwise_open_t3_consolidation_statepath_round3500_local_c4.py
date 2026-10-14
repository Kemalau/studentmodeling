from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/watch_layerwise_open_t3_consolidation_statepath_round3500_local_c4.sh"
).read_text()


def test_consolidation_full_c4_is_frozen_audited_and_read_only() -> None:
    for fragment in (
        "state_path/lr5e-7_start13000_stop3500_",
        "checkpoint-round-3500",
        "assert observed==receipt['sha256']",
        "c4_stage_a_gate_pairs_seed42_n150_v1.json",
        "--num-pairs 150",
        "--state-architecture layerwise",
        "--state-token-alias '<|vision_pad|>'",
        "local_gpu_readonly_preflight.lock",
        "open-t3-consolidation-statepath-lr5e7",
    ):
        assert fragment in SCRIPT
    for forbidden in ("amlt run", "optimizer.step", "--train-jsonl"):
        assert forbidden not in SCRIPT
