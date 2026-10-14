import json
from pathlib import Path

import pytest

from scripts.select_state8_strong1_widen8_bridge import score


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def fixture(root: Path, slug: str = "lr5e7_seed58") -> None:
    train_job = f"state8-strong1-widen8-bridge-{slug.replace('_', '-')}"
    if slug == "lr5e7_seed58":
        train_job = "state8-strong1-widen8-bridge-lr5e7-seed58"
        eval_job = "state8-strong1-widen8-eval-lr5e7-seed58"
    else:
        train_job = "state8-strong1-widen8-bridge-lr1e6-seed59"
        eval_job = "state8-strong1-widen8-eval-lr1e6-seed59"
    artifacts = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    groups = {
        "predict_correctness/a": {"accuracy": 0.7, "n": 25},
        "predict_response/a": {"math_equivalence": 0.4, "n": 25},
        "predict_prospective_error/a": {"joint_math_equivalence": 0.5, "n": 25},
        "predict_error_persistence/a": {
            "accuracy": 0.6,
            "macro_f1": 0.5,
            "positive_recall": 0.4,
            "n": 25,
        },
    }
    base = root / "train_download/x" / train_job
    write(
        base / "eval/formal_screen/metrics.json",
        {
            "artifact_sha256": artifacts,
            "state_shape": [64, 8, 5120],
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "metrics_by_group": groups,
        },
    )
    write(
        base / "metrics/immutable_receipt.json",
        {
            "status": "complete",
            "state_shape": [64, 8, 5120],
            "raw_history_visible_to_decoder": False,
            "sha256": artifacts,
            "remote_root": "/mnt/amulet/test",
        },
    )
    write(
        base / "metrics/taxonomy_teacher_curriculum.json",
        {
            "semantic_judge_authoritative": True,
            "exact_label_metric_authoritative": False,
            "output_schema": "label_diagnosis_evidence_v1",
        },
    )
    t3_groups = {
        "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
            "llm_judge_n": 25,
            "llm_judge_exact_shortcuts": 0,
            "llm_judge_accuracy": 0.3,
            "natural_schema_valid_rate": 1.0,
            "llm_judge_schema_valid_n": 25,
            "llm_judge_raw_semantic_accuracy": 0.3,
        },
        "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {
            "llm_judge_n": 25,
            "llm_judge_exact_shortcuts": 0,
            "llm_judge_accuracy": 0.6,
            "natural_schema_valid_rate": 1.0,
            "llm_judge_schema_valid_n": 25,
            "llm_judge_raw_semantic_accuracy": 0.6,
        },
        "diagnose_misconception/malrule/synthetic_same_template_open_generation": {
            "llm_judge_n": 25,
            "llm_judge_exact_shortcuts": 0,
            "llm_judge_accuracy": 0.6,
            "natural_schema_valid_rate": 1.0,
            "llm_judge_schema_valid_n": 25,
            "llm_judge_raw_semantic_accuracy": 0.6,
        },
    }
    write(
        root / slug / "t3_semantic/metrics.json",
        {"judge_backend": "trapi", "judge_errors": 0,
         "judge_rubric": "t3-semantic-equivalence-v2-substantive-schema",
         "metrics_by_group": t3_groups},
    )
    diag_cells = {"swap_to_donor": {"semantic_equivalence_rate": 0.7}}
    write(
        root / slug / "c4_semantic/summary.json",
        {
            "pass": True,
            "semantic_D_specific": 0.3,
            "same_rule_recipient_retention": 0.8,
            "noop_recipient_retention": 0.9,
            "random_matches_either_stable_target_rate": 0.1,
            "cells": diag_cells,
            "generation": {"artifact_sha256": artifacts},
        },
    )
    answer = root / "eval_download/x" / eval_job
    write(
        answer / "eval/answer_c4_seed42_n150/recurrent_causal_gate.json",
        {
            "artifact_sha256": artifacts,
            "state_shape": [64, 8, 5120],
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "summary": {
                "scientific_gate_pass": True,
                "D_specific": 0.3,
                "clean": {
                    "swap": {"flip_to_A_rate": 0.5},
                    "random": {"flip_to_A_rate": 0.2},
                    "samemal": {"flip_to_A_rate": 0.05},
                    "noop": {"flip_to_A_rate": 0.0},
                },
            },
        },
    )


def test_joint_gate_requires_every_scientific_axis(tmp_path: Path) -> None:
    fixture(tmp_path)
    value = score("lr5e7_seed58", tmp_path)
    assert value["qualified"] is True
    path = tmp_path / "lr5e7_seed58/c4_semantic/summary.json"
    broken = json.loads(path.read_text())
    broken["pass"] = False
    write(path, broken)
    assert score("lr5e7_seed58", tmp_path)["qualified"] is False


def test_joint_gate_rejects_checkpoint_binding_drift(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = next(tmp_path.glob("eval_download/**/recurrent_causal_gate.json"))
    broken = json.loads(path.read_text())
    broken["artifact_sha256"]["adapter_model"] = "c" * 64
    write(path, broken)
    with pytest.raises(ValueError, match="Checkpoint binding drift"):
        score("lr5e7_seed58", tmp_path)


def test_joint_gate_rejects_stale_semantic_rubric(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / "lr5e7_seed58/t3_semantic/metrics.json"
    broken = json.loads(path.read_text())
    broken["judge_rubric"] = "t3-semantic-equivalence-v1"
    write(path, broken)
    with pytest.raises(ValueError, match="Stale or non-substantive"):
        score("lr5e7_seed58", tmp_path)
