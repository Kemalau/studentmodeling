from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_state_kd_local_preflight_is_read_only_and_hash_bound() -> None:
    script = (ROOT / "scripts/watch_layerwise_state_kd_lr5e6_round3250_local_preflight.sh").read_text()
    required = (
        "checkpoint-round-3250",
        "eedi_full/state_kd/lr5e-6",
        "assert observed==receipt['sha256']",
        "--state-architecture layerwise",
        "--num-pairs 150",
        "--max-samples-per-group 10",
        "not_a_formal_gate",
    )
    assert all(fragment in script for fragment in required)
    forbidden = ("amlt run", "optimizer.step", "--train-jsonl", "scientific_gate_pass\"")
    assert not any(fragment in script for fragment in forbidden)
