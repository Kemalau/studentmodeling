#!/usr/bin/env python3
"""Joint final gate for state8 tasks, semantic T3, and multiseed C4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FROZEN_GROUP_SUPPORT = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 894,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 300,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 300,
    "predict_correctness/dbe_kt22/temporal_warm_item_binary": 225,
    "predict_correctness/eedi_response/profile_heldout_binary": 984,
    "predict_correctness/foundationalassist/temporal_warm_item_binary": 1882,
    "predict_correctness/malrule/synthetic_cross_template_binary": 300,
    "predict_correctness/malrule/synthetic_same_template_binary": 300,
    "predict_correctness/xes3g5m/temporal_warm_item_binary": 7222,
    "predict_error_persistence/dbe_kt22/strict_exact_recurrence_binary": 12,
    "predict_error_persistence/foundationalassist/strict_exact_recurrence_binary": 964,
    "predict_prospective_error/dbe_kt22/delta3_closed_choice_generation/delta=3": 200,
    "predict_prospective_error/foundationalassist/delta_open_generation/delta=1": 1485,
    "predict_prospective_error/foundationalassist/delta_open_generation/delta=3": 1474,
    "predict_prospective_error/foundationalassist/delta_open_generation/delta=5": 1441,
    "predict_response/dbe_kt22/closed_choice_generation": 223,
    "predict_response/eedi_response/closed_choice_generation": 984,
    "predict_response/foundationalassist/structured_math_open_generation": 1886,
    "predict_response/malrule/synthetic_cross_template_open_generation": 300,
    "predict_response/malrule/synthetic_same_template_open_generation": 300,
}
REFERENCES = {
    "predict_correctness": ("macro_f1", 0.6869980398379085, 0.90),
    "predict_response": ("math_equivalence", 0.5182778229082047, 0.90),
    "predict_prospective_error": ("joint_math_equivalence", 0.7667391304347826, 0.90),
    # Strict T6 is strongly imbalanced, so accuracy can pass after a complete
    # collapse to the majority class.  Reuse the frozen full-support reference
    # and retention floor from the task-retention experiments, and separately
    # require non-zero positive recall below.
    "predict_error_persistence": ("macro_f1", 0.5959866676584252, 0.85),
}
DIAGNOSIS_FIRST_VARIANTS = {
    "strong1_rescue_taxonomy_joint_seed61",
    "strong1_midpoint_diagnosis_first_seed63",
    "strong1_exact_diagnosis_curriculum_seed65",
    "strong1_exact_rescue_semantic_preserve_seed66",
    "strong1_exact_rescue_causal_taskanchor_seed67",
    "strong1_exact_rescue_retention_preserve_seed68",
    "diagnosis_first_matched_seed67",
    "same_question_updater_mse1_seed3072",
    "same_question_full_mse1_seed3071",
    "same_question_repair_promoted",
    "same_question_repair_checkpoint_promoted",
    "same_question_curriculum_promoted",
}
SAME_QUESTION_VARIANTS = {
    "same_question_updater_mse1_seed3072",
    "same_question_full_mse1_seed3071",
    "same_question_repair_promoted",
    "same_question_repair_checkpoint_promoted",
    "same_question_curriculum_promoted",
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def weighted(groups: dict[str, Any], task: str, metric: str) -> dict[str, Any]:
    rows = [value for name, value in groups.items() if name.startswith(task + "/")]
    if not rows or any(metric not in row for row in rows):
        raise ValueError(f"Missing {metric} for {task}")
    support = sum(int(row["n"]) for row in rows)
    score = sum(float(row[metric]) * int(row["n"]) for row in rows) / support
    return {"metric": metric, "score": score, "support": support}


def validate(
    formal_path: Path,
    c4_path: Path,
    semantic_gate_path: Path,
    semantic_metrics_path: Path,
    diagnosis_c4_path: Path,
    final_fit_receipt_path: Path,
    eedi_same_question_c4_path: Path | None = None,
) -> dict[str, Any]:
    formal = read(formal_path)
    c4 = read(c4_path)
    semantic = read(semantic_gate_path)
    judge = read(semantic_metrics_path)
    diagnosis_c4 = read(diagnosis_c4_path)
    receipt = read(final_fit_receipt_path)
    eedi_same_question_c4 = (
        read(eedi_same_question_c4_path)
        if eedi_same_question_c4_path is not None
        else None
    )
    same_question_variant = str(receipt.get("selected_replay_variant") or "") in SAME_QUESTION_VARIANTS
    groups = formal.get("metrics_by_group") or {}
    formal_artifacts = formal.get("artifact_sha256") or {}
    expected_artifacts = {
        "adapter_model": receipt.get("adapter_sha256"),
        "recurrent_state": receipt.get("recurrent_state_sha256"),
    }
    diagnosis_rows = diagnosis_c4.get("rows") or []
    checks: dict[str, bool] = {
        "final_fit_receipt": receipt.get("status") == "final_fit_complete"
        and receipt.get("stage") == "state8_qualified_unified_final_fit_v1"
        and receipt.get("one_epoch_exact") is True
        and int(receipt.get("real_examples", -1)) == 259845,
        "final_fit_state_contract": receipt.get("state_shape") == [64, 8, 5120]
        and receipt.get("raw_history_visible_to_decoder") is False
        and receipt.get("persistent_history_kv_cache") is False,
        "final_fit_strict_teacher_contract": receipt.get("history_training_mode")
        == "strict_state_only"
        and receipt.get("history_dropout_counts") == {"1.0": 259845}
        and float(receipt.get("online_kd_weight", -1)) == 1.0
        and float(receipt.get("label_auxiliary_weight", -1))
        == (
            1.0
            if receipt.get("selected_replay_variant") in DIAGNOSIS_FIRST_VARIANTS
            else 4.0
        )
        and (
            receipt.get("selected_replay_variant") not in DIAGNOSIS_FIRST_VARIANTS
            or (
                receipt.get("diagnosis_field_order")
                == ["diagnosis", "evidence", "label"]
                and receipt.get("canonical_label_role")
                == "auxiliary_ordinary_weight"
            )
        )
        and float(receipt.get("trajectory_weight", -1)) in {0.0, 0.1, 1.0},
        "final_fit_diagnosis_causal_supervision": receipt.get("causal_supervision")
        == ["prospective_answer", "diagnosis_natural_schema"]
        and receipt.get("causal_target_kind_counts")
        == (
            {
                "diagnosis_natural_schema": 2302,
                "prospective_answer": 2102,
            }
            if receipt.get("selected_replay_variant") in SAME_QUESTION_VARIANTS
            else {
                "diagnosis_natural_schema": 2102,
                "prospective_answer": 2102,
            }
        )
        and receipt.get("causal_output_schema")
        == (
            "diagnosis_evidence_label_v1"
            if receipt.get("selected_replay_variant")
            in ({"diagnosis_first_matched_seed67"} | SAME_QUESTION_VARIANTS)
            else "label_diagnosis_evidence_v1"
        )
        and receipt.get("canonical_label_auxiliary_supervision") is True,
        "final_fit_artifact_binding": all(
            isinstance(value, str)
            and len(value) == 64
            and formal_artifacts.get(key) == value
            for key, value in expected_artifacts.items()
        ),
        "formal_stage": formal.get("stage") == "state8_final_formal_merge_v1",
        "state_shape": formal.get("state_shape") == [64, 8, 5120],
        "history_closed": formal.get("raw_history_visible_to_decoder") is False
        and formal.get("persistent_history_kv_cache") is False,
        "formal_support": int(formal.get("sampled_rows", -1)) == 21676,
        "all_groups": set(groups) == set(FROZEN_GROUP_SUPPORT),
        "group_support": all(
            name in groups and int(groups[name].get("n", -1)) == support
            for name, support in FROZEN_GROUP_SUPPORT.items()
        ),
        "T6_secondary_support": sum(
            int(value.get("n", 0))
            for name, value in groups.items()
            if name.startswith("predict_error_persistence/")
        )
        == 976,
        "semantic_T3": semantic.get("stage") == "final_T3_semantic_judge"
        and semantic.get("pass") is True
        and int(semantic.get("support", -1)) == 1494,
        "semantic_backend": judge.get("judge_backend") == "trapi",
        "semantic_complete": judge.get("generation_rows") == 1494
        and judge.get("judged_rows") == 1494
        and judge.get("judge_errors") == 0,
        "semantic_no_exact_shortcuts": all(
            int(value.get("llm_judge_exact_shortcuts", -1)) == 0
            for value in (judge.get("metrics_by_group") or {}).values()
        ),
        "C4_stage": c4.get("stage") == "state8_promotion_c4_multiseed_v1",
        "C4_replication": c4.get("replication_gate_pass") is True,
        "C4_specificity": float(c4.get("mean_D_specific", -1)) >= 0.15,
        "C4_seed_support": {int(row["pair_seed"]) for row in c4.get("rows", [])}
        == {13, 42, 97},
        "C4_random_control": sum(
            float(row.get("random", 1.0)) <= 0.25 for row in c4.get("rows", [])
        )
        >= 2,
        "C4_noop_control": sum(
            float(row.get("noop", 1.0)) <= 0.05 for row in c4.get("rows", [])
        )
        >= 2,
        "C4_same_rule_control": sum(
            float(row.get("same_rule", 1.0)) <= 0.10
            for row in c4.get("rows", [])
        )
        >= 2,
        "diagnosis_C4_stage": diagnosis_c4.get("stage")
        == "state8_closed_diagnosis_c4_semantic_multiseed_v1",
        "diagnosis_C4_artifact_binding": len(diagnosis_rows) == 3
        and {
            int((row.get("generation") or {}).get("pair_seed", -1))
            for row in diagnosis_rows
        }
        == {13, 42, 97}
        and all(
            (row.get("generation") or {}).get("artifact_sha256")
            == expected_artifacts
            and int((row.get("generation") or {}).get("num_pairs", -1)) == 150
            and (row.get("generation") or {}).get("state_shape")
            == [64, 8, 5120]
            and (row.get("generation") or {}).get(
                "raw_history_visible_to_decoder"
            )
            is False
            and (row.get("generation") or {}).get("persistent_history_kv_cache")
            is False
            for row in diagnosis_rows
        ),
        "diagnosis_C4_semantic_authority": diagnosis_c4.get(
            "semantic_judge_authoritative"
        )
        is True
        and diagnosis_c4.get("exact_label_role")
        == "auxiliary_only_not_used_for_gate",
        "diagnosis_C4_replication": diagnosis_c4.get("pass") is True
        and int(diagnosis_c4.get("seed_passes", -1)) >= 2,
        "diagnosis_C4_specificity": float(
            diagnosis_c4.get("mean_semantic_D_specific", -1)
        )
        >= 0.15,
        "diagnosis_C4_swap": float(diagnosis_c4.get("mean_swap_to_donor", -1))
        >= 0.50,
        "diagnosis_C4_random_control": float(
            diagnosis_c4.get("mean_random_matches_either", 1)
        )
        <= 0.25,
        "eedi_same_question_C4_required": (
            eedi_same_question_c4 is not None
            if same_question_variant
            else eedi_same_question_c4 is None
        ),
    }
    if eedi_same_question_c4 is not None:
        eedi_generation = eedi_same_question_c4.get("generation") or {}
        eedi_cells = eedi_same_question_c4.get("cells") or {}
        checks.update(
            {
                "eedi_same_question_C4_artifact_binding":
                    eedi_generation.get("artifact_sha256") == expected_artifacts
                    and int(eedi_generation.get("num_pairs", -1)) == 150
                    and eedi_generation.get("state_shape") == [64, 8, 5120],
                "eedi_same_question_C4_closed_query":
                    eedi_generation.get("same_question_query_identical") is True
                    and eedi_generation.get("current_item_visible_to_decoder") is True
                    and eedi_generation.get("student_response_visible_to_decoder") is False
                    and eedi_generation.get("raw_history_visible_to_decoder") is False
                    and eedi_generation.get("persistent_history_kv_cache") is False,
                "eedi_same_question_C4_semantic_authority":
                    eedi_same_question_c4.get("stage")
                    == "state8_eedi_same_question_diagnosis_c4_semantic_final_v1"
                    and eedi_same_question_c4.get("confirmatory") is True
                    and eedi_same_question_c4.get("frozen_panel") is True
                    and eedi_same_question_c4.get("exploratory_only") is False
                    and eedi_same_question_c4.get("semantic_judge_authoritative") is True
                    and eedi_same_question_c4.get("exact_label_role")
                    == "auxiliary_only_not_used_for_gate"
                    and eedi_same_question_c4.get("pass") is True,
                "eedi_same_question_C4_specificity": float(
                    eedi_same_question_c4.get("semantic_D_specific", -1)
                ) >= 0.15,
                "eedi_same_question_C4_swap": float(
                    (eedi_cells.get("swap_to_donor") or {}).get(
                        "semantic_equivalence_rate", -1
                    )
                ) >= 0.50,
                "eedi_same_question_C4_random_control": float(
                    eedi_same_question_c4.get(
                        "random_matches_either_stable_target_rate", 1
                    )
                ) <= 0.25,
                "eedi_same_question_C4_noop_control": float(
                    eedi_same_question_c4.get("noop_recipient_retention", -1)
                ) >= 0.75,
                "eedi_same_question_C4_same_rule_control": float(
                    eedi_same_question_c4.get("same_rule_recipient_retention", -1)
                ) >= 0.65,
            }
        )
    performance: dict[str, Any] = {}
    for task, (metric, reference, floor) in REFERENCES.items():
        row = weighted(groups, task, metric)
        row.update(
            {
                "reference": reference,
                "minimum_retention_ratio": floor,
                "retention_ratio": row["score"] / reference,
            }
        )
        row["pass"] = row["retention_ratio"] >= floor
        performance[task] = row
    t6_recall = weighted(
        groups, "predict_error_persistence", "positive_recall"
    )
    t6_recall["pass"] = t6_recall["score"] > 0.0
    performance["predict_error_persistence"]["positive_recall"] = t6_recall[
        "score"
    ]
    performance["predict_error_persistence"]["positive_recall_pass"] = (
        t6_recall["pass"]
    )
    checks["T6_positive_recall"] = t6_recall["pass"]
    checks["non_T3_task_retention"] = all(row["pass"] for row in performance.values())
    # Canonical-label/exact metrics remain in `formal` as auxiliary diagnostics.
    # They are deliberately absent from `checks`; T3 authority is TRAPI semantic judgment.
    return {
        "stage": "state8_complete_unified_model_final_gate_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "performance": performance,
        "T3_semantic": semantic,
        "C4": c4,
        "diagnosis_C4": diagnosis_c4,
        "eedi_same_question_C4": eedi_same_question_c4,
        "artifact_sha256": formal_artifacts,
        "final_fit_receipt": {
            "stage": receipt.get("stage"),
            "one_epoch_exact": receipt.get("one_epoch_exact"),
            "real_examples": receipt.get("real_examples"),
            "adapter_sha256": receipt.get("adapter_sha256"),
            "recurrent_state_sha256": receipt.get("recurrent_state_sha256"),
        },
        "T3_exact_match_role": "auxiliary_only_not_used_for_gate",
        "T6_role": "secondary_but_gated_macro_f1_retention_and_positive_recall",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--c4", type=Path, required=True)
    parser.add_argument("--semantic-gate", type=Path, required=True)
    parser.add_argument("--semantic-metrics", type=Path, required=True)
    parser.add_argument("--diagnosis-c4", type=Path, required=True)
    parser.add_argument("--final-fit-receipt", type=Path, required=True)
    parser.add_argument("--eedi-same-question-c4", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.formal,
        args.c4,
        args.semantic_gate,
        args.semantic_metrics,
        args.diagnosis_c4,
        args.final_fit_receipt,
        args.eedi_same_question_c4,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
