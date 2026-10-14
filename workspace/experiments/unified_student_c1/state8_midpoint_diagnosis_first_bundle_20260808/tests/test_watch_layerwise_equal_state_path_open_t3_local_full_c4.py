from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_c4_reuses_hash_audited_checkpoint_and_frozen_protocol() -> None:
    text = (
        ROOT / "scripts/watch_layerwise_equal_state_path_open_t3_local_full_c4.sh"
    ).read_text(encoding="utf-8")
    assert "checkpoint-round-3250" in text
    assert "audit.get('sha256')==receipt.get('sha256')" in text
    assert "c4_stage_a_gate_pairs_seed42_n150_v1.json" in text
    assert "--num-pairs 150" in text
    assert "--state-token-alias '<|vision_pad|>'" in text
    assert "state-architecture layerwise" in text
    assert "local_gpu_readonly_preflight.lock" in text
