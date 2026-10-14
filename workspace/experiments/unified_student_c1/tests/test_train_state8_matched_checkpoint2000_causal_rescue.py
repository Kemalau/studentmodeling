from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/train_state8_matched_checkpoint2000_causal_rescue.py").read_text()


def test_matched_rescue_binds_diagnosis_first_state_only_contract() -> None:
    for fragment in (
        'OUTPUT_SCHEMA = "diagnosis_evidence_label_v1"',
        'namespace.diagnosis_field_order = ["diagnosis", "evidence", "label"]',
        'namespace.canonical_label_role = "auxiliary_ordinary_weight"',
        'namespace.semantic_judge_authoritative = True',
        'namespace.exact_match_authoritative = False',
        'namespace.raw_history_visible_to_decoder = False',
        'namespace.final_inference_route = "strict_state_only"',
        'output_dir / "diagnosis_first_causal_repair.json"',
    ):
        assert fragment in SOURCE


def test_matched_rescue_composes_both_screen_arms() -> None:
    assert 'mode == "semantic_preserve"' in SOURCE
    assert 'else anchored' in SOURCE
    assert 'mode == "causal_taskanchor"' in SOURCE
