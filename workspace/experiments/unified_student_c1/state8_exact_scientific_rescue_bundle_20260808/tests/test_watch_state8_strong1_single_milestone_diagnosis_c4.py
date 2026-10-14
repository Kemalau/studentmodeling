from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_live_diagnosis_c4_curve_is_read_only_hash_bound_and_semantic() -> None:
    source = (
        ROOT / "scripts/watch_state8_strong1_single_milestone_diagnosis_c4.sh"
    ).read_text()
    for fragment in (
        "steps=(1000 2000 3000 4000 5000)",
        "checkpoint_completion.json",
        "receipt['sha256']==observed",
        "config['state_token_count']==8",
        "config['final_inference_route']=='strict_state_only'",
        "taxonomy_label_contract",
        "local_gpu_readonly_preflight.lock",
        "min_free_mib=49152",
        "--query-gpu=memory.free",
        '[[ -f "$root/complete" && -s "$root/summary.json"',
        "completed diagnosis-C4 milestone binding drift",
        "evaluate_recurrent_diagnosis_c4_taxonomy.py",
        "--expected-adapter-sha256",
        "--expected-recurrent-state-sha256",
        "judge_formal_t3_predictions.py",
        "summarize_state8_diagnosis_c4.py",
        "semantic_judge_authoritative",
        "exact_label_role",
        "diagnostic_only",
    ):
        assert fragment in source
    assert "amlt run" not in source
    assert "amlt cancel" not in source
    assert "az storage blob delete" not in source
    assert "--query-compute-apps=pid" not in source
