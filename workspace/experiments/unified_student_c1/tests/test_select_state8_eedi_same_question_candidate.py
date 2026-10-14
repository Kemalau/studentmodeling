from scripts.select_state8_eedi_same_question_candidate import select


def c4():
    return {
        "pass": True,
        "semantic_judge_authoritative": True,
        "exact_label_role": "auxiliary_only_not_used_for_gate",
        "semantic_D_specific": 0.4,
        "noop_recipient_retention": 0.8,
        "same_rule_recipient_retention": 0.8,
        "random_matches_either_stable_target_rate": 0.1,
        "cells": {"swap_to_donor": {"semantic_equivalence_rate": 0.7}},
        "generation": {
            "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
        },
    }


def summary(stage="state8_eedi_same_question_causal_repair_semantic_summary_v1"):
    groups = {
        "predict_correctness/x": {
            "accuracy": 0.7, "macro_f1": 0.65, "n": 25
        },
        "predict_response/x": {"math_equivalence": 0.4, "n": 25},
        "predict_prospective_error/x": {"joint_math_equivalence": 0.5, "n": 25},
        "predict_error_persistence/x": {
            "accuracy": 0.5, "macro_f1": 0.4, "positive_recall": 0.2, "n": 25
        },
    }
    semantic = {
        "judge_backend": "trapi", "judge_errors": 0,
        "metrics_by_group": {
            "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
                "llm_judge_accuracy": 0.4, "llm_judge_n": 25, "natural_schema_valid_rate": 1.0
            },
            "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {
                "llm_judge_accuracy": 0.6, "llm_judge_n": 25, "natural_schema_valid_rate": 1.0
            },
            "diagnose_misconception/malrule/synthetic_same_template_open_generation": {
                "llm_judge_accuracy": 0.6, "llm_judge_n": 25, "natural_schema_valid_rate": 1.0
            },
        },
    }
    return {
        "stage": stage, "status": "complete", "semantic_judge_authoritative": True,
        "exact_match_authoritative": False, "formal_metrics": {"metrics_by_group": groups},
        "t3_semantic": semantic, "malrule_diagnosis_c4": c4(),
        "eedi_same_question_diagnosis_c4": c4(),
    }


def test_joint_selection_requires_all_unified_and_both_c4_gates():
    result = select({
        "updater": summary(),
        "full": summary("state8_eedi_same_question_full_mse1_semantic_summary_v1"),
    })
    assert result["pass"] is True
    assert result["selected"]["qualified"] is True
    assert result["answer_C4_confirmation_pending"] is True


def test_failed_eedi_c4_blocks_candidate():
    updater = summary()
    full = summary("state8_eedi_same_question_full_mse1_semantic_summary_v1")
    updater["eedi_same_question_diagnosis_c4"]["pass"] = False
    full["eedi_same_question_diagnosis_c4"]["pass"] = False
    result = select({"updater": updater, "full": full})
    assert result["pass"] is False
    assert result["selected"] is None
    assert all(
        "eedi_same_question_C4_pass" in row["failed_checks"]
        for row in result["candidates"].values()
    )


def test_dynamic_repair_candidate_can_be_selected_with_audited_metadata():
    value = summary("state8_eedi_same_question_repair_semantic_summary_v1")
    value["candidate_metadata"] = {
        "job": "state8-eedi-repair-semantic-binding-seed3201",
        "seed": 3201,
        "remote_root": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed3201_taskanchor_source_anchor_"
            "full_mse1_eedi_same_question_v1_semantic_binding_seed3201"
        ),
        "finalfit_variant": "same_question_repair_promoted",
        "repair_branch": "semantic_binding",
        "source_anchor_mode": "full_mse1",
    }
    result = select({"repair_semantic_binding_seed3201": value})
    assert result["pass"] is True
    selected = result["selected"]
    assert selected["finalfit_variant"] == "same_question_repair_promoted"
    assert selected["repair_branch"] == "semantic_binding"
    assert selected["artifact_sha256"] == {
        "adapter_model": "a" * 64,
        "recurrent_state": "b" * 64,
    }
