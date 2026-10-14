from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_task_gates_are_hash_bound_and_match_remote_protocol() -> None:
    text = (
        ROOT / "scripts/watch_layerwise_equal_state_path_open_t3_local_task_gates.sh"
    ).read_text(encoding="utf-8")
    assert "audit.get('sha256')==receipt.get('sha256')" in text
    assert "c4.get('all_pairs_n')==150" in text
    assert "c4.get('scientific_gate_pass') is True" in text
    assert "--max-samples-per-group 25" in text
    assert "--tasks predict_error_persistence" in text
    assert "--state-token-alias '<|vision_pad|>'" in text
    assert "--state-architecture layerwise" in text
    assert "local_gpu_readonly_preflight.lock" in text
