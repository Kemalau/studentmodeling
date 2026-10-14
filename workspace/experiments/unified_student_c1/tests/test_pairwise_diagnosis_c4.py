import json

from scripts.build_pairwise_diagnosis_c4_judge_panel import build_panel
from scripts.summarize_state8_diagnosis_c4_pairwise import summarize


def generation(pair, condition, diagnosis):
    return {
        "pair_idx": pair,
        "condition": condition,
        "decode_prompt": "state plus fixed query",
        "raw_output": (
            f"<diagnosis>{diagnosis}</diagnosis>"
            "<evidence></evidence><label>x</label>"
        ),
    }


def test_pairwise_panel_uses_native_diagnosis_without_schema_gate():
    rows = [
        generation(1, condition, "native rule" if condition != "swap" else "donor rule")
        for condition in ("native", "noop", "samemal", "random", "swap")
    ]
    predictions, evaluation, receipt = build_panel(rows)
    assert len(predictions) == len(evaluation) == 4
    assert receipt["eligible_native_pairs"] == 1
    assert all(row["target"] == "<misconception>native rule</misconception>" for row in predictions)
    assert {row["condition"] for row in predictions} == {"noop", "samemal", "random", "swap"}


def test_pairwise_summary_separates_invariance_from_gold_correctness():
    judged = []
    values = {"noop": True, "samemal": True, "random": False, "swap": False}
    for condition, equivalent in values.items():
        judged.append(
            {
                "condition": condition,
                "reference_role": "native_output",
                "llm_judge_semantic_equivalent": equivalent,
            }
        )
    result = summarize(
        judged,
        {"eligible_native_pairs": 1, "excluded_native_pairs": []},
    )
    assert result["noop_native_output_invariance"] == 1.0
    assert result["same_rule_native_output_invariance"] == 1.0
    assert result["random_native_output_invariance"] == 0.0
    assert result["swap_native_output_invariance"] == 0.0
    assert result["same_rule_vs_swap_separation"] == 1.0
