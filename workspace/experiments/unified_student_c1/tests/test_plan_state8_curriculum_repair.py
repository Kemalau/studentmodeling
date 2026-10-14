from scripts.plan_state8_curriculum_repair import plan


def summary(
    arm: str,
    step: int,
    *,
    t3: float = 0.10,
    task: float = 0.20,
    c4_pass: bool = False,
    specificity: float = 0.0,
    identity: float = 0.40,
):
    artifacts = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    diagnosis = {
        "semantic_judge_authoritative": True,
        "exact_label_role": "auxiliary_only_not_used_for_gate",
        "pass": c4_pass,
        "semantic_D_specific": specificity,
        "random_matches_either_stable_target_rate": 0.50,
        "noop_recipient_retention": identity,
        "same_rule_recipient_retention": identity,
        "cells": {"swap_to_donor": {"semantic_equivalence_rate": specificity}},
        "generation": {"artifact_sha256": artifacts},
    }
    return {
        "stage": "state8_curriculum_checkpoint_unified_semantic_summary_v1",
        "status": "complete",
        "diagnostic_only": True,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "arm": arm,
        "checkpoint_step": step,
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "artifact_sha256": artifacts,
        "task_metrics": {
            "predict_correctness": {"score": task},
            "predict_response": {"score": task},
            "predict_prospective_error": {"score": task},
            "predict_error_persistence": {"score": task},
        },
        "T6_positive_recall": {"score": 0.0},
        "T3_semantic": {
            "support": 75,
            "accuracy": t3,
            "eedi_accuracy": t3,
            "metrics_by_group": {
                "malrule/synthetic_cross_template": {
                    "llm_judge_n": 25,
                    "llm_judge_accuracy": t3,
                    "natural_schema_valid_rate": 1.0,
                },
                "malrule/synthetic_same_template": {
                    "llm_judge_n": 25,
                    "llm_judge_accuracy": t3,
                    "natural_schema_valid_rate": 1.0,
                },
            },
        },
        "answer_C4": {"pass": c4_pass, "D_specific": specificity},
        "malrule_diagnosis_C4": diagnosis,
        "eedi_same_question_C4": diagnosis,
    }


def matrix(**kwargs):
    return [
        summary(arm, step, **kwargs)
        for arm in ("generic", "current_query")
        for step in (2000, 3000, 4000, 5000)
    ]


def test_failed_curriculum_matrix_creates_bounded_parallel_repairs():
    result = plan(matrix())
    assert result["scientific_floors_unchanged"] is True
    assert result["raw_history_visible_to_decoder"] is False
    assert 1 <= result["parallel_gpu_jobs"] <= 3
    assert {branch["source"]["family"] for branch in result["branches"]} == {
        "curriculum"
    }
    assert "semantic_binding" in {branch["name"] for branch in result["branches"]}
    assert "specificity_triplet" in {branch["name"] for branch in result["branches"]}
    assert "balanced_continuation" in {
        branch["name"] for branch in result["branches"]
    }
    assert "identity_controls" not in {
        branch["name"] for branch in result["branches"]
    }
    specificity = next(
        branch for branch in result["branches"]
        if branch["name"] == "specificity_triplet"
    )
    assert specificity["training_arm"] == "full"


def test_qualified_curriculum_checkpoint_is_not_sent_to_repair():
    values = matrix(t3=0.80, task=0.80, c4_pass=True, specificity=0.60, identity=0.90)
    for value in values:
        value["T6_positive_recall"]["score"] = 0.50
    try:
        plan(values)
    except ValueError as error:
        assert "qualified" in str(error)
    else:
        raise AssertionError("A qualified curriculum checkpoint should be promoted")


def test_one_good_identity_checkpoint_does_not_trigger_identity_repair():
    values = matrix(identity=0.20)
    values[0]["malrule_diagnosis_C4"]["noop_recipient_retention"] = 0.90
    values[0]["malrule_diagnosis_C4"]["same_rule_recipient_retention"] = 0.90
    values[0]["eedi_same_question_C4"]["noop_recipient_retention"] = 0.90
    values[0]["eedi_same_question_C4"]["same_rule_recipient_retention"] = 0.90
    result = plan(values)
    assert result["symptoms"]["identity_controls"] is False


def test_schema_only_failure_routes_to_semantic_binding():
    values = matrix(t3=0.80, task=0.80, c4_pass=True, specificity=0.60, identity=0.90)
    for value in values:
        value["T6_positive_recall"]["score"] = 0.50
        for group in value["T3_semantic"]["metrics_by_group"].values():
            group["natural_schema_valid_rate"] = 0.80
    result = plan(values)
    assert result["symptoms"]["semantic_capacity"] is True
    assert "T3_schema" in result["failed_checks"]
    assert "semantic_binding" in {row["name"] for row in result["branches"]}
