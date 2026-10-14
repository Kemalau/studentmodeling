from scripts.plan_state8_eedi_same_question_repair import plan


def summary(arm: str, step: int, *, t3=0.1, specificity=0.0, random=0.5,
            identity=0.4, synthetic=0.6):
    c4 = {
        "semantic_D_specific": specificity,
        "random_matches_either_stable_target_rate": random,
        "noop_recipient_retention": identity,
        "same_rule_recipient_retention": identity,
        "cells": {
            "swap_to_donor": {"semantic_equivalence_rate": max(0.0, specificity)},
            "native_to_recipient": {"semantic_equivalence_rate": identity},
        },
    }
    return {
        "stage": "state8_eedi_same_question_checkpoint_semantic_summary_v1",
        "status": "complete",
        "diagnostic_only": True,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "arm": arm,
        "checkpoint_step": step,
        "T3_eedi_semantic": t3,
        "T3_synthetic_cross_semantic": synthetic,
        "T3_synthetic_same_semantic": synthetic,
        "malrule_diagnosis_c4": c4,
        "eedi_same_question_diagnosis_c4": c4,
    }


def matrix(**kwargs):
    return [summary(arm, step, **kwargs) for arm in ("updater", "full") for step in (20, 100)]


def test_failed_matrix_derives_three_orthogonal_bounded_branches():
    result = plan(matrix())
    assert result["scientific_floors_unchanged"] is True
    assert result["raw_history_visible_to_decoder"] is False
    assert [branch["name"] for branch in result["branches"]] == [
        "semantic_binding", "specificity_triplet", "identity_controls"
    ]
    assert result["parallel_gpu_jobs"] == 3
    assert all(branch["training_arm"] in {"updater", "full"} for branch in result["branches"])


def test_late_erosion_is_detected_and_early_source_is_preferred():
    values = []
    for arm in ("updater", "full"):
        values.append(summary(arm, 20, t3=0.19, specificity=0.09, random=0.45, identity=0.55))
        values.append(summary(arm, 100, t3=0.01, specificity=-0.1, random=0.8, identity=0.1))
    result = plan(values)
    assert result["symptoms"]["late_training_erosion"] is True
    assert all(branch["source"]["checkpoint_step"] == 20 for branch in result["branches"])


def test_planner_rejects_a_matrix_that_already_has_a_viable_checkpoint():
    values = matrix(t3=0.4, specificity=0.4, random=0.1, identity=0.8)
    try:
        plan(values)
    except ValueError as error:
        assert "viable" in str(error)
    else:
        raise AssertionError("Viable matrix should not enter repair planning")
