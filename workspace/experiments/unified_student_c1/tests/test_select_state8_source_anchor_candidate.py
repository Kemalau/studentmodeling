from scripts.select_state8_source_anchor_candidate import evaluate


def candidate() -> dict:
    groups = {
        "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
            "llm_judge_accuracy": 0.4,
            "natural_schema_valid_rate": 1.0,
        },
        "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {
            "llm_judge_accuracy": 0.7,
            "natural_schema_valid_rate": 1.0,
        },
        "diagnose_misconception/malrule/synthetic_same_template_open_generation": {
            "llm_judge_accuracy": 0.7,
            "natural_schema_valid_rate": 1.0,
        },
    }
    formal = {
        "predict_correctness/a/x": {"n": 1, "accuracy": 0.7},
        "predict_response/a/x": {"n": 1, "math_equivalence": 0.5},
        "predict_prospective_error/a/x": {"n": 1, "joint_math_equivalence": 0.6},
        "predict_error_persistence/a/x": {
            "n": 1, "accuracy": 0.6, "macro_f1": 0.6, "positive_recall": 0.5,
        },
    }
    diagnosis = {
        "pass": True,
        "semantic_D_specific": 0.3,
        "random_matches_either_stable_target_rate": 0.1,
        "noop_recipient_retention": 0.8,
        "same_rule_recipient_retention": 0.8,
        "cells": {"swap_to_donor": {"semantic_equivalence_rate": 0.7}},
    }
    return {
        "stage": "state8_source_anchor_candidate_semantic_summary_v1",
        "status": "complete",
        "state_token_count": 8,
        "raw_history_visible_to_decoder": False,
        "final_inference_route": "strict_state_only",
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "formal_metrics": formal,
        "t3_metrics_by_group": groups,
        "diagnosis_c4": diagnosis,
    }


def test_diagnostic_selection_requires_all_task_and_semantic_c4_floors() -> None:
    result = evaluate(candidate())
    assert result["diagnostic_qualified"] is True
    broken = candidate()
    broken["diagnosis_c4"]["random_matches_either_stable_target_rate"] = 0.4
    assert evaluate(broken)["diagnostic_qualified"] is False
