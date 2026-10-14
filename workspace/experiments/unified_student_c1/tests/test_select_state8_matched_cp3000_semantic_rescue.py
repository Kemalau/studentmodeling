from __future__ import annotations

import json
from pathlib import Path

from scripts.select_state8_matched_cp3000_semantic_rescue import select


def formal_metrics() -> dict:
    return {
        "predict_correctness/a": {"accuracy": 0.7, "n": 25},
        "predict_response/a": {"math_equivalence": 0.4, "n": 25},
        "predict_prospective_error/a": {"joint_math_equivalence": 0.5, "n": 25},
        "predict_error_persistence/a": {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "positive_recall": 0.2,
            "n": 25,
        },
    }


def write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_joint_promotion_passes_only_bound_strict_state_candidate(tmp_path: Path) -> None:
    artifacts = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    groups = {
        "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
            "llm_judge_n": 25,
            "llm_judge_accuracy": 0.4,
            "natural_schema_valid_rate": 1.0,
        },
        "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {
            "llm_judge_n": 25,
            "llm_judge_accuracy": 0.6,
            "natural_schema_valid_rate": 1.0,
        },
        "diagnose_misconception/malrule/synthetic_same_template_open_generation": {
            "llm_judge_n": 25,
            "llm_judge_accuracy": 0.6,
            "natural_schema_valid_rate": 1.0,
        },
    }
    summary = {
        "stage": "state8_matched_checkpoint3000_semantic_rescue_summary_v1",
        "status": "complete",
        "source_checkpoint_step": 3000,
        "state_token_count": 8,
        "raw_history_visible_to_decoder": False,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "artifact_sha256": artifacts,
        "formal_metrics": formal_metrics(),
        "t3_support": 75,
        "t3_metrics_by_group": groups,
        "answer_c4": {
            "scientific_gate_pass": True,
            "D_specific": 0.3,
            "swap": 0.6,
            "random": 0.2,
            "noop": 0.0,
            "same_rule": 0.1,
        },
        "diagnosis_c4": {
            "pass": True,
            "semantic_D_specific": 0.3,
            "noop_recipient_retention": 0.8,
            "same_rule_recipient_retention": 0.7,
            "random_matches_either_stable_target_rate": 0.2,
            "cells": {"swap_to_donor": {"semantic_equivalence_rate": 0.6}},
        },
    }
    receipt = {
        "status": "complete",
        "source_checkpoint_step": 3000,
        "rescue_mode": "semantic_preserve",
        "state_token_count": 8,
        "history_dropout_schedule": [1.0],
        "raw_history_visible_to_decoder": False,
        "final_inference_route": "strict_state_only",
        "canonical_label_role": "auxiliary_ordinary_weight",
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "artifact_sha256": artifacts,
    }
    result = select(write(tmp_path / "summary.json", summary), write(tmp_path / "receipt.json", receipt))
    assert result["pass"] is True
    assert result["selected"]["artifact_sha256"] == artifacts


def test_joint_promotion_rejects_history_leakage(tmp_path: Path) -> None:
    summary = {
        "stage": "state8_matched_checkpoint3000_semantic_rescue_summary_v1",
        "status": "complete",
        "source_checkpoint_step": 3000,
        "state_token_count": 8,
        "raw_history_visible_to_decoder": True,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
    }
    receipt = {"status": "complete", "source_checkpoint_step": 3000}
    try:
        select(write(tmp_path / "summary.json", summary), write(tmp_path / "receipt.json", receipt))
    except ValueError as error:
        assert "history" in str(error).lower()
    else:
        raise AssertionError("history leakage was accepted")
