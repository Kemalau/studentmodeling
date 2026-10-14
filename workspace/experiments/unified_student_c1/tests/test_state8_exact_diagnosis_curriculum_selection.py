from __future__ import annotations

from pathlib import Path

import json

import pytest

from scripts.select_state8_exact_diagnosis_curriculum import JOB, score
from scripts.select_state8_strong1_widen8_bridge import REQUIRED_T3_RUBRIC, T3_GROUPS


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root: Path) -> None:
    artifacts = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    artifact = root / "download" / "experiment" / JOB
    formal_groups = {
        "predict_correctness/a": {"n": 10, "accuracy": 0.8},
        "predict_response/a": {"n": 10, "math_equivalence": 0.7},
        "predict_prospective_error/a": {"n": 10, "joint_math_equivalence": 0.7},
        "predict_error_persistence/a": {
            "n": 10, "accuracy": 0.7, "macro_f1": 0.65, "positive_recall": 0.6,
        },
    }
    state_contract = {
        "artifact_sha256": artifacts,
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
    }
    write(artifact / "eval/formal_t3_300/metrics.json", {
        **state_contract, "metrics_by_group": formal_groups,
    })
    write(artifact / "metrics/publish_receipt.json", {
        "stage": "state8_stateonly_repair_pilot_publish_v1",
        "train_entrypoint": "scripts/train_state8_diagnosis_first_causal_repair.py",
        "artifact_sha256": artifacts,
    })
    write(artifact / "eval/answer_c4_150/recurrent_causal_gate.json", {
        **state_contract,
        "summary": {
            "scientific_gate_pass": True, "D_specific": 0.5,
            "clean": {"samemal": {"flip_to_A_rate": 0.0}, "noop": {"flip_to_A_rate": 0.0}},
        },
    })
    write(artifact / "metrics/immutable_receipt.json", {
        "status": "complete", "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "history_dropout_counts": {
            "0.0": 1000, "0.25": 1000, "0.5": 1000, "0.75": 1000, "1.0": 1000,
        },
    })
    teacher_recipe = {
        "field_order": ["diagnosis", "evidence", "label"],
        "canonical_label_role": "auxiliary_ordinary_weight",
        "label_auxiliary_weight": 1.0,
        "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    write(artifact / "metrics/diagnosis_first_teacher_curriculum.json", teacher_recipe)
    write(artifact / "metrics/diagnosis_first_causal_repair.json", {
        key: value for key, value in teacher_recipe.items() if key != "history_dropout_schedule"
    })
    write(root / "t3_semantic/metrics.json", {
        "judge_backend": "trapi", "judge_errors": 0, "judge_rubric": REQUIRED_T3_RUBRIC,
        "metrics_by_group": {
            name: {
                "llm_judge_n": 10, "llm_judge_exact_shortcuts": 0,
                "llm_judge_accuracy": 0.8, "natural_schema_valid_rate": 1.0,
            }
            for name in T3_GROUPS
        },
    })
    write(root / "c4_semantic/summary.json", {
        "generation": {"artifact_sha256": artifacts}, "pass": True,
        "semantic_D_specific": 0.5,
        "cells": {"swap_to_donor": {"semantic_equivalence_rate": 0.8}},
        "same_rule_recipient_retention": 0.8, "noop_recipient_retention": 0.9,
        "random_matches_either_stable_target_rate": 0.1,
    })


def set_publish_entrypoint(root: Path, entrypoint: str) -> None:
    artifact = next((root / "download").glob(f"**/{JOB}"))
    path = artifact / "metrics/publish_receipt.json"
    value = json.loads(path.read_text())
    value["train_entrypoint"] = entrypoint
    write(path, value)


def test_exact_diagnosis_candidate_is_jointly_gated_and_promotable() -> None:
    selector = (ROOT / "scripts/select_state8_exact_diagnosis_curriculum.py").read_text()
    for fragment in (
        "state8_exact_diagnosis_curriculum_joint_promotion_v1",
        "exact_diagnosis_curriculum_seed65",
        "strong1_exact_diagnosis_curriculum_seed65",
        '"diagnosis", "evidence", "label"',
        "history_dropout_counts",
        "llm_judge_exact_shortcuts",
        "answer_C4_pass",
        "diagnosis_C4_pass",
        "exact_match_authoritative",
        "semantic_judge_authoritative",
    ):
        assert fragment in selector

    watcher = (ROOT / "scripts/watch_state8_exact_diagnosis_pipeline.sh").read_text()
    assert "select_state8_exact_diagnosis_curriculum.py" in watcher
    assert "scientific_gate_pass" in watcher
    assert "scientific_gate_failed" in watcher


def test_exact_diagnosis_candidate_reaches_full_fit_and_final_eval() -> None:
    supervisor = (ROOT / "scripts/supervise_state8_strong1_widen8_complete.sh").read_text()
    launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
    publisher = (ROOT / "scripts/publish_state8_final_checkpoint.py").read_text()
    formal = (ROOT / "scripts/run_state8_final_formal_shard.sh").read_text()
    causal = (ROOT / "scripts/run_state8_final_c4.sh").read_text()
    validator = (ROOT / "scripts/validate_state8_final_eval.py").read_text()
    for text in (supervisor, launcher, publisher, formal, causal, validator):
        assert "strong1_exact_diagnosis_curriculum_seed65" in text
    assert "state8_exact_diagnosis_curriculum_v1/selection.json" in supervisor
    assert "state8_exact_diagnosis_curriculum_joint_promotion_v1" in launcher
    assert "state8_stateonly_repair_seed65_trajectory01_exact_teacher_curriculum_c4_seed65_v1" in launcher
    assert "train_state8_diagnosis_first_causal_repair.py" in launcher
    assert "STATE8_REPAIR_LABEL_WEIGHT=1.0" in launcher
    assert "EXPECTED_HISTORY_MODE=strict_state_only" in launcher


def test_exact_diagnosis_selector_passes_complete_semantic_contract(tmp_path: Path) -> None:
    fixture(tmp_path)
    candidate = score(tmp_path)
    assert candidate["qualified"] is True
    assert candidate["checks"]["T3_eedi_semantic"] == pytest.approx(0.8)
    assert candidate["checks"]["diagnosis_C4_D_specific"] == pytest.approx(0.5)
    assert candidate["finalfit_variant"] == "strong1_exact_diagnosis_curriculum_seed65"


def test_exact_diagnosis_selector_accepts_hash_audited_exact_resume_wrapper(
    tmp_path: Path,
) -> None:
    fixture(tmp_path)
    set_publish_entrypoint(
        tmp_path,
        "scripts/train_state8_diagnosis_first_causal_repair_exact_resume.py",
    )
    candidate = score(tmp_path)
    assert candidate["qualified"] is True


def test_exact_diagnosis_selector_rejects_unapproved_entrypoint(tmp_path: Path) -> None:
    fixture(tmp_path)
    set_publish_entrypoint(tmp_path, "scripts/unrelated_training_wrapper.py")
    with pytest.raises(ValueError, match="entrypoint drift"):
        score(tmp_path)


def test_exact_diagnosis_selector_fails_closed_on_history_or_exact_shortcut(tmp_path: Path) -> None:
    fixture(tmp_path)
    artifact = next((tmp_path / "download").glob(f"**/{JOB}"))
    formal = json.loads((artifact / "eval/formal_t3_300/metrics.json").read_text())
    formal["raw_history_visible_to_decoder"] = True
    write(artifact / "eval/formal_t3_300/metrics.json", formal)
    with pytest.raises(ValueError, match="History leakage"):
        score(tmp_path)

    fixture(tmp_path)
    judged = json.loads((tmp_path / "t3_semantic/metrics.json").read_text())
    next(iter(judged["metrics_by_group"].values()))["llm_judge_exact_shortcuts"] = 1
    write(tmp_path / "t3_semantic/metrics.json", judged)
    with pytest.raises(ValueError, match="T3 authority drift"):
        score(tmp_path)
