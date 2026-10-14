from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_final_eval import (  # noqa: E402
    FROZEN_GROUP_SUPPORT,
    REFERENCES,
    validate,
)


def dump(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def inputs(tmp_path: Path, random_rates: list[float]):
    groups = {}
    for name, n in FROZEN_GROUP_SUPPORT.items():
        task = name.split("/", 1)[0]
        row = {"n": n}
        if task in REFERENCES:
            metric, reference, _ = REFERENCES[task]
            row[metric] = reference
        if task == "predict_error_persistence":
            row["positive_recall"] = 0.5
        groups[name] = row
    formal = {
        "stage": "state8_final_formal_merge_v1",
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "sampled_rows": 21676,
        "metrics_by_group": groups,
        "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }
    rows = []
    for seed, random in zip((13, 42, 97), random_rates, strict=True):
        rows.append(
            {
                "pair_seed": seed,
                "D_specific": 0.2,
                "random": random,
                "noop": 0.0,
                "same_rule": 0.05,
                "swap": random + 0.2,
                "scientific_gate_pass": random <= 0.25,
            }
        )
    c4 = {
        "stage": "state8_promotion_c4_multiseed_v1",
        "replication_gate_pass": sum(row["scientific_gate_pass"] for row in rows) >= 2,
        "mean_D_specific": 0.2,
        "rows": rows,
    }
    semantic = {"stage": "final_T3_semantic_judge", "pass": True, "support": 1494}
    judge_groups = {
        name: {"llm_judge_exact_shortcuts": 0}
        for name in FROZEN_GROUP_SUPPORT
        if name.startswith("diagnose_misconception/")
    }
    judge = {
        "judge_backend": "trapi",
        "generation_rows": 1494,
        "judged_rows": 1494,
        "judge_errors": 0,
        "metrics_by_group": judge_groups,
    }
    diagnosis_c4 = {
        "stage": "state8_closed_diagnosis_c4_semantic_multiseed_v1",
        "semantic_judge_authoritative": True,
        "exact_label_role": "auxiliary_only_not_used_for_gate",
        "seed_passes": 3,
        "mean_semantic_D_specific": 0.4,
        "mean_swap_to_donor": 0.7,
        "mean_random_matches_either": 0.1,
        "pass": True,
        "rows": [
            {
                "generation": {
                    "pair_seed": seed,
                    "num_pairs": 150,
                    "state_shape": [64, 8, 5120],
                    "raw_history_visible_to_decoder": False,
                    "persistent_history_kv_cache": False,
                    "artifact_sha256": {
                        "adapter_model": "a" * 64,
                        "recurrent_state": "b" * 64,
                    },
                }
            }
            for seed in (13, 42, 97)
        ],
    }
    receipt = {
        "status": "final_fit_complete",
        "stage": "state8_qualified_unified_final_fit_v1",
        "one_epoch_exact": True,
        "real_examples": 259845,
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "causal_supervision": ["prospective_answer", "diagnosis_natural_schema"],
        "causal_target_kind_counts": {
            "diagnosis_natural_schema": 2102,
            "prospective_answer": 2102,
        },
        "causal_output_schema": "label_diagnosis_evidence_v1",
        "canonical_label_auxiliary_supervision": True,
        "history_training_mode": "strict_state_only",
        "history_dropout_counts": {"1.0": 259845},
        "online_kd_weight": 1.0,
        "label_auxiliary_weight": 4.0,
        "trajectory_weight": 0.0,
        "adapter_sha256": "a" * 64,
        "recurrent_state_sha256": "b" * 64,
    }
    return (
        dump(tmp_path / "formal.json", formal),
        dump(tmp_path / "c4.json", c4),
        dump(tmp_path / "semantic.json", semantic),
        dump(tmp_path / "judge.json", judge),
        dump(tmp_path / "diagnosis_c4.json", diagnosis_c4),
        dump(tmp_path / "receipt.json", receipt),
    )


def test_final_gate_explicitly_accepts_two_of_three_random_controls(tmp_path: Path) -> None:
    value = validate(*inputs(tmp_path, [0.1, 0.2, 0.3]))
    assert value["checks"]["C4_random_control"] is True
    assert value["pass"] is True


def test_final_gate_rejects_random_target_migration(tmp_path: Path) -> None:
    value = validate(*inputs(tmp_path, [0.3, 0.4, 0.2]))
    assert value["checks"]["C4_random_control"] is False
    assert value["pass"] is False


def test_final_gate_rejects_artifact_not_bound_to_final_fit(tmp_path: Path) -> None:
    paths = list(inputs(tmp_path, [0.1, 0.2, 0.1]))
    receipt = json.loads(paths[-1].read_text())
    receipt["adapter_sha256"] = "c" * 64
    paths[-1].write_text(json.dumps(receipt))
    value = validate(*paths)
    assert value["checks"]["final_fit_artifact_binding"] is False
    assert value["pass"] is False


def test_final_gate_accepts_task_anchor_trajectory_weight(tmp_path: Path) -> None:
    paths = list(inputs(tmp_path, [0.1, 0.2, 0.1]))
    receipt = json.loads(paths[-1].read_text())
    receipt["trajectory_weight"] = 1.0
    paths[-1].write_text(json.dumps(receipt))
    value = validate(*paths)
    assert value["checks"]["final_fit_strict_teacher_contract"] is True
    assert value["pass"] is True


def test_final_gate_rejects_t6_majority_class_collapse(tmp_path: Path) -> None:
    paths = list(inputs(tmp_path, [0.1, 0.2, 0.1]))
    formal = json.loads(paths[0].read_text())
    for name, row in formal["metrics_by_group"].items():
        if name.startswith("predict_error_persistence/"):
            row["positive_recall"] = 0.0
    paths[0].write_text(json.dumps(formal))
    value = validate(*paths)
    assert value["checks"]["T6_positive_recall"] is False
    assert value["pass"] is False


def test_final_gate_rejects_t6_macro_f1_retention_failure(tmp_path: Path) -> None:
    paths = list(inputs(tmp_path, [0.1, 0.2, 0.1]))
    formal = json.loads(paths[0].read_text())
    for name, row in formal["metrics_by_group"].items():
        if name.startswith("predict_error_persistence/"):
            row["macro_f1"] = 0.1
    paths[0].write_text(json.dumps(formal))
    value = validate(*paths)
    t6 = value["performance"]["predict_error_persistence"]
    assert t6["retention_ratio"] < t6["minimum_retention_ratio"]
    assert t6["pass"] is False
    assert value["pass"] is False


def test_final_gate_rejects_diagnosis_c4_artifact_drift(tmp_path: Path) -> None:
    paths = list(inputs(tmp_path, [0.1, 0.2, 0.1]))
    diagnosis = json.loads(paths[4].read_text())
    diagnosis["rows"][1]["generation"]["artifact_sha256"][
        "recurrent_state"
    ] = "c" * 64
    paths[4].write_text(json.dumps(diagnosis))
    value = validate(*paths)
    assert value["checks"]["diagnosis_C4_artifact_binding"] is False
    assert value["pass"] is False
