from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_live_t3_curve_is_read_only_hash_bound_strict_and_semantic_v2() -> None:
    source = (
        ROOT / "scripts/watch_state8_strong1_single_milestone_t3.sh"
    ).read_text()
    for fragment in (
        "steps=(1000 2000 3000 4000 5000)",
        "checkpoint_completion.json",
        "receipt['sha256']==observed",
        "config['state_token_count']==8",
        "config['final_inference_route']=='strict_state_only'",
        "local_gpu_readonly_preflight.lock",
        "min_free_mib=49152",
        "--query-gpu=memory.free",
        '[[ -f "$root/complete" && -s "$root/summary.json"',
        "completed T3 milestone binding drift",
        "evaluate_recurrent_formal_protocols_taxonomy.py",
        "--tasks diagnose_misconception",
        "--sources eedi_mining malrule",
        "--judge-backend trapi",
        "t3-semantic-equivalence-v2-substantive-schema",
        "llm_judge_raw_semantic_accuracy",
        "llm_judge_schema_valid_n",
        "exact_match_authoritative':False",
    ):
        assert fragment in source
    assert "amlt run" not in source
    assert "amlt cancel" not in source
    assert "az storage blob delete" not in source
    assert "--query-compute-apps=pid" not in source
