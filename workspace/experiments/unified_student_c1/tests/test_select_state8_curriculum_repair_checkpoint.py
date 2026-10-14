from copy import deepcopy

import pytest

from scripts.select_state8_curriculum_repair_checkpoint import evaluate, select


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


def summary(seed=6201, branch="semantic_binding"):
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
        "arm": f"repair_{branch}_seed{seed}", "checkpoint_step": 500,
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
        "answer_C4": {"pass": False, "D_specific": 0.1},
        "malrule_diagnosis_C4": diagnosis(artifact),
        "eedi_same_question_C4": diagnosis(artifact),
    }


def test_repair_checkpoint_requires_shared_task_and_dual_c4_gates():
    result = evaluate(summary())
    assert result["qualified"] is True
    assert result["artifact_layout"] == "repair_checkpoint_snapshot"
    assert result["finalfit_variant"] == "same_question_repair_checkpoint_promoted"
    broken = summary()
    broken["T6_positive_recall"]["score"] = 0.0
    result = evaluate(broken)
    assert result["qualified"] is False
    assert "T6_positive_recall" in result["failed_checks"]


def test_selection_requires_complete_cp3000_cp4000_six_branch_matrix():
    values = [
        summary(5201, "semantic_binding"),
        summary(5202, "specificity_triplet"),
        summary(5203, "balanced_continuation"),
        summary(6201, "semantic_binding"),
        summary(6202, "specificity_triplet"),
        summary(6203, "balanced_continuation"),
    ]
    result = select(values)
    assert result["pass"] is True
    assert result["selected"]["remote_root"].endswith("/checkpoint-500")
    by_seed = {row["repair_seed"]: row for row in result["candidates"]}
    assert by_seed[5201]["source_checkpoint_step"] == 4000
    assert by_seed[6201]["source_checkpoint_step"] == 3000
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
