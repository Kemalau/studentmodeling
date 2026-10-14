from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step1250_c4_is_read_only_hash_bound_and_diagnostic_only() -> None:
    source = (ROOT / "scripts/run_state8_strong1_single_step1250_c4.sh").read_text()
    for fragment in (
        "step=1250",
        "receipt['sha256']==observed",
        "counts=={0.0:1000,0.25:250}",
        "config['final_inference_route']=='strict_state_only'",
        "local_gpu_readonly_preflight.lock",
        "evaluate_recurrent_causal_gate.py",
        "--num-pairs 50",
        "excluded_from_promotion",
        "rm -f \"$checkpoint/recurrent_state.pt\"",
    ):
        assert fragment in source
    assert "amlt run" not in source
    assert "amlt cancel" not in source
