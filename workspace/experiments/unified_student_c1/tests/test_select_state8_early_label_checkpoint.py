from copy import deepcopy

import pytest

from scripts.select_state8_early_label_checkpoint import select


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


def summary(seed, branch, step):
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


def early(seed: int, branch: str, step: int, *, qualified: bool = True):
    value = summary(seed, branch, step)
    if not qualified:
        value = deepcopy(value)
        value["T6_positive_recall"]["score"] = 0.0
    return value


def test_adaptive_pair_selects_only_after_shared_gates() -> None:
    result = select([
        early(8301, "early_label_low", 20),
        early(8302, "early_label_high", 20, qualified=False),
    ])
    assert result["pass"] is True
    assert result["adaptive_checkpoint_step"] == 20
    assert result["selected"]["repair_branch"] == "early_label_low"
    assert result["selected"]["source_history_mode"].endswith(
        "early_label_binding"
    )


def test_adaptive_pair_fails_closed_and_rejects_mixed_steps() -> None:
    result = select([
        early(8301, "early_label_low", 100, qualified=False),
        early(8302, "early_label_high", 100, qualified=False),
    ])
    assert result["pass"] is False
    with pytest.raises(ValueError, match="step drift"):
        select([
            early(8301, "early_label_low", 20),
            early(8302, "early_label_high", 100),
        ])


def test_adaptive_selector_projects_legacy_checks_to_public_v2_floors() -> None:
    result = select([
        early(8301, "early_label_low", 20),
        early(8302, "early_label_high", 20),
    ])
    assert result["pass"] is True
    floors = result["selected"]["floors"]
    assert "T4_joint_math_equivalence" in floors
    assert "T5_macro_f1" in floors
    assert "T5_positive_recall" in floors
    assert "T5_joint_math_equivalence" not in floors
    assert "T6_macro_f1" not in floors
