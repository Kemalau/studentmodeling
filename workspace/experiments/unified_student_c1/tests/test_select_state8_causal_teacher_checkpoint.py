from copy import deepcopy

import pytest

from scripts.select_state8_causal_teacher_checkpoint import evaluate, select


def diagnosis(artifact):
    return {
        "pass": True,
        "semantic_judge_authoritative": True,
        "exact_label_role": "auxiliary_only_not_used_for_gate",
        "semantic_D_specific": 0.25,
        "random_matches_either_stable_target_rate": 0.10,
        "noop_recipient_retention": 0.85,
        "same_rule_recipient_retention": 0.80,
        "cells": {"swap_to_donor": {"semantic_equivalence_rate": 0.65}},
        "generation": {"artifact_sha256": artifact},
    }


def summary(seed=8201, branch="causal_teacher_low", step=100):
    artifact = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    groups = {
        "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
            "llm_judge_n": 25, "llm_judge_accuracy": 0.45,
            "natural_schema_valid_rate": 1.0,
        },
        "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {
            "llm_judge_n": 25, "llm_judge_accuracy": 0.60,
            "natural_schema_valid_rate": 1.0,
        },
        "diagnose_misconception/malrule/synthetic_same_template_open_generation": {
            "llm_judge_n": 25, "llm_judge_accuracy": 0.60,
            "natural_schema_valid_rate": 1.0,
        },
    }
    return {
        "stage": "state8_curriculum_checkpoint_unified_semantic_summary_v1",
        "status": "complete", "diagnostic_only": True,
        "arm": f"repair_{branch}_seed{seed}", "checkpoint_step": step,
        "state_shape": [64, 8, 5120], "artifact_sha256": artifact,
        "raw_history_visible_to_decoder": False,
        "semantic_judge_authoritative": True, "exact_match_authoritative": False,
        "task_metrics": {
            "predict_correctness": {"score": 0.65},
            "predict_response": {"score": 0.30},
            "predict_prospective_error": {"score": 0.45},
            "predict_error_persistence": {"score": 0.40},
        },
        "T6_positive_recall": {"score": 0.20},
        "T3_semantic": {
            "accuracy": 0.55, "eedi_accuracy": 0.45, "support": 75,
            "metrics_by_group": groups,
        },
        "malrule_diagnosis_C4": diagnosis(artifact),
        "eedi_same_question_C4": diagnosis(artifact),
    }


def test_causal_teacher_checkpoint_requires_shared_task_and_dual_c4_gates():
    result = evaluate(summary())
    assert result["qualified"] is True
    assert result["repair_branch"] == "causal_teacher_low"
    broken = summary()
    broken["T6_positive_recall"]["score"] = 0.0
    result = evaluate(broken)
    assert result["qualified"] is False
    assert "T5_positive_recall" in result["failed_checks"]


def test_selection_requires_low_high_full_milestone_matrix():
    values = [
        summary(8201, "causal_teacher_low", 20),
        summary(8201, "causal_teacher_low", 100),
        summary(8201, "causal_teacher_low", 240),
        summary(8201, "causal_teacher_low", 500),
        summary(8202, "causal_teacher_high", 20),
        summary(8202, "causal_teacher_high", 100),
        summary(8202, "causal_teacher_high", 240),
        summary(8202, "causal_teacher_high", 500),
    ]
    result = select(values)
    assert result["pass"] is True
    assert len(result["candidates"]) == 8
    with pytest.raises(ValueError, match="matrix drift"):
        select(values[:-1])


def test_c4_artifacts_must_match_checkpoint_hashes():
    value = summary()
    value["eedi_same_question_C4"] = deepcopy(value["eedi_same_question_C4"])
    value["eedi_same_question_C4"]["generation"]["artifact_sha256"] = {
        "adapter_model": "c" * 64, "recurrent_state": "d" * 64,
    }
    with pytest.raises(ValueError, match="artifact binding drift"):
        evaluate(value)
