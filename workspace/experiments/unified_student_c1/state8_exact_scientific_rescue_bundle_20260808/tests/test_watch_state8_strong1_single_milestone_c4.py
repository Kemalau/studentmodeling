from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_live_milestone_probe_is_read_only_hash_bound_and_strict_state_only() -> None:
    source = (
        ROOT / "scripts/watch_state8_strong1_single_milestone_c4.sh"
    ).read_text()
    for fragment in (
        "steps=(250 1000 2000 3000 4000 5000)",
        "progress is (step-1)/4999",
        "checkpoint_completion.json",
        "receipt['sha256']==observed",
        "config['state_token_count']==8",
        "config['final_inference_route']=='strict_state_only'",
        "local_gpu_readonly_preflight.lock",
        "min_free_mib=49152",
        "--query-gpu=memory.free",
        '[[ -f "$root/complete" && -s "$root/summary.json"',
        "completed answer-C4 milestone binding drift",
        "heartbeat_pid",
        "while true; do heartbeat; sleep 30; done",
        "evaluate_recurrent_causal_gate.py",
        "--num-pairs 50",
        "diagnostic_only",
    ):
        assert fragment in source
    assert "amlt cancel" not in source
    assert "amlt run" not in source
    assert "--query-compute-apps=pid" not in source
