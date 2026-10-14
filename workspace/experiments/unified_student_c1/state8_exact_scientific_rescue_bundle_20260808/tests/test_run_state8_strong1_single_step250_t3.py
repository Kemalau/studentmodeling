from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step250_t3_probe_is_hash_bound_strict_and_semantically_judged() -> None:
    source = (ROOT / "scripts/run_state8_strong1_single_step250_t3.sh").read_text()
    for fragment in (
        "r['sha256']==a['sha256']",
        "local_gpu_readonly_preflight.lock",
        "evaluate_recurrent_formal_protocols_taxonomy.py",
        "--tasks diagnose_misconception",
        "--sources eedi_mining malrule",
        "x['raw_history_visible_to_decoder'] is False",
        "--judge-backend trapi",
        "t3-semantic-equivalence-v2-substantive-schema",
        "semantic['judge_rubric']==rubric",
        "llm_judge_exact_shortcuts",
        "llm_judge_raw_semantic_accuracy",
        "llm_judge_schema_valid_n",
        "exact_match_authoritative':False",
    ):
        assert fragment in source
    assert "amlt run" not in source


def test_step250_t3_summary_can_be_rebuilt_after_checkpoint_cleanup() -> None:
    source = (ROOT / "scripts/run_state8_strong1_single_step250_t3.sh").read_text()
    assert 'if [[ ! -s "$formal/metrics.json" ]]; then' in source
    preflight = source.split('if [[ ! -s "$formal/metrics.json" ]]; then', 1)[0]
    assert '"$checkpoint/adapter/adapter_model.safetensors"' not in preflight
    assert '"$checkpoint/recurrent_state.pt"' not in preflight
