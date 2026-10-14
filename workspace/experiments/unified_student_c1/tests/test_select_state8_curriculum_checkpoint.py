from scripts.select_state8_curriculum_checkpoint import evaluate, select


def diagnosis(artifact):
    return {
        "pass": True,
        "semantic_judge_authoritative": True,
        "exact_label_role": "auxiliary_only_not_used_for_gate",
        "semantic_D_specific": 0.2,
        "random_matches_either_stable_target_rate": 0.1,
        "noop_recipient_retention": 0.8,
        "same_rule_recipient_retention": 0.8,
        "cells": {"swap_to_donor": {"semantic_equivalence_rate": 0.6}},
        "generation": {"artifact_sha256": artifact},
    }


def summary(arm="generic", step=2000):
    artifact = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    groups = {
        "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
            "llm_judge_n": 25, "llm_judge_accuracy": 0.4,
            "natural_schema_valid_rate": 1.0,
        },
        "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {
            "llm_judge_n": 25, "llm_judge_accuracy": 0.6,
            "natural_schema_valid_rate": 1.0,
        },
        "diagnose_misconception/malrule/synthetic_same_template_open_generation": {
            "llm_judge_n": 25, "llm_judge_accuracy": 0.6,
            "natural_schema_valid_rate": 1.0,
        },
    }
    return {
        "stage": "state8_curriculum_checkpoint_unified_semantic_summary_v1",
        "status": "complete", "diagnostic_only": True,
        "arm": arm, "checkpoint_step": step,
        "state_shape": [64, 8, 5120], "artifact_sha256": artifact,
        "raw_history_visible_to_decoder": False,
        "semantic_judge_authoritative": True, "exact_match_authoritative": False,
        "task_metrics": {
            "predict_correctness": {"score": 0.6},
            "predict_response": {"score": 0.3},
            "predict_prospective_error": {"score": 0.5},
            "predict_error_persistence": {"score": 0.4},
        },
        "T6_positive_recall": {"score": 0.2},
        "T3_semantic": {
            "accuracy": 0.5, "eedi_accuracy": 0.4, "support": 75,
            "metrics_by_group": groups,
        },
        "answer_C4": {"pass": True, "D_specific": 0.2},
        "malrule_diagnosis_C4": diagnosis(artifact),
        "eedi_same_question_C4": diagnosis(artifact),
    }


def test_evaluate_requires_every_hard_gate():
    assert evaluate(summary())["qualified"] is True
    broken = summary()
    broken["T6_positive_recall"]["score"] = 0.0
    result = evaluate(broken)
    assert result["qualified"] is False
    assert "T6_positive_recall" in result["failed_checks"]


def test_selection_requires_complete_eight_checkpoint_matrix():
    values = [summary(arm, step) for arm in ("generic", "current_query") for step in (2000, 3000, 4000, 5000)]
    result = select(values)
    assert result["pass"] is True
    assert result["selected"]["finalfit_variant"] == "same_question_curriculum_promoted"
    try:
        select(values[:-1])
    except ValueError as error:
        assert "matrix drift" in str(error)
    else:
        raise AssertionError("Incomplete curriculum matrix was accepted")
