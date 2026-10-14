#!/usr/bin/env python3
"""Promote the checkpoint-3000 semantic rescue only after its joint gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from select_state8_strong1_widen8_bridge import task_mean, task_weighted
except ImportError:
    from scripts.select_state8_strong1_widen8_bridge import task_mean, task_weighted


SLUG = "diagnosis_first_matched_cp3000_semantic_seed3066"
REMOTE_ROOT = (
    "/mnt/amulet/studentmodeling/diagnostics/"
    "state8_stateonly_repair_seed3066_trajectory01_"
    "matched_cp3000_semantic_preserve_v1"
)
FINALFIT_VARIANT = SLUG
FINALFIT_JOB = "state8-matched-cp3000-semantic-seed3066-unified-finalfit-seed42"


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def select(summary_path: Path, rescue_receipt_path: Path) -> dict[str, Any]:
    summary = read(summary_path)
    receipt = read(rescue_receipt_path)
    if summary.get("stage") != "state8_matched_checkpoint3000_semantic_rescue_summary_v1":
        raise ValueError("Semantic-rescue summary stage drift")
    if summary.get("status") != "complete" or summary.get("source_checkpoint_step") != 3000:
        raise ValueError("Semantic-rescue summary is incomplete")
    if summary.get("semantic_judge_authoritative") is not True:
        raise ValueError("Semantic judge is not authoritative")
    if summary.get("exact_match_authoritative") is not False:
        raise ValueError("Exact match authority drift")
    if summary.get("state_token_count") != 8:
        raise ValueError("State-token count drift")
    if summary.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Decoder history leakage")
    if receipt.get("status") != "complete" or receipt.get("source_checkpoint_step") != 3000:
        raise ValueError("Rescue receipt is incomplete")
    if receipt.get("rescue_mode") != "semantic_preserve":
        raise ValueError("Rescue mode drift")
    if receipt.get("state_token_count") != 8 or receipt.get("history_dropout_schedule") != [1.0]:
        raise ValueError("Strict-state-only rescue contract drift")
    if receipt.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Rescue receipt exposes history")
    if receipt.get("final_inference_route") != "strict_state_only":
        raise ValueError("Final inference route drift")
    if receipt.get("canonical_label_role") != "auxiliary_ordinary_weight":
        raise ValueError("Canonical-label role drift")
    if receipt.get("semantic_judge_authoritative") is not True:
        raise ValueError("Receipt semantic authority drift")
    if receipt.get("exact_match_authoritative") is not False:
        raise ValueError("Receipt exact-match authority drift")

    artifacts = summary.get("artifact_sha256") or {}
    if set(artifacts) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Summary artifact binding drift")
    if receipt.get("artifact_sha256") != artifacts:
        raise ValueError("Summary/receipt artifact mismatch")
    if not all(isinstance(value, str) and len(value) == 64 for value in artifacts.values()):
        raise ValueError("Invalid artifact SHA256")

    formal = summary.get("formal_metrics") or {}
    diagnosis = summary.get("diagnosis_c4") or {}
    answer = summary.get("answer_c4") or {}
    groups = summary.get("t3_metrics_by_group") or {}
    if len(groups) != 3 or int(summary.get("t3_support", -1)) != 75:
        raise ValueError("T3 semantic screen coverage drift")
    eedi_rows = [row for name, row in groups.items() if "/eedi_mining/" in name]
    synthetic_rows = [row for name, row in groups.items() if "/malrule/" in name]
    if len(eedi_rows) != 1 or len(synthetic_rows) != 2:
        raise ValueError("T3 semantic group drift")
    if any(int(row.get("llm_judge_n", -1)) != 25 for row in groups.values()):
        raise ValueError("T3 judge support drift")
    t3_eedi = float(eedi_rows[0]["llm_judge_accuracy"])
    t3_synthetic = sum(float(row["llm_judge_accuracy"]) for row in synthetic_rows) / 2
    t3_schema = sum(float(row["natural_schema_valid_rate"]) for row in groups.values()) / 3

    checks = {
        "T1": task_mean(formal, "predict_correctness/", "accuracy"),
        "T2": task_mean(formal, "predict_response/", "math_equivalence"),
        "T3_eedi_semantic": t3_eedi,
        "T3_synthetic_semantic": t3_synthetic,
        "T3_schema": t3_schema,
        "T5": task_mean(formal, "predict_prospective_error/", "joint_math_equivalence"),
        "T6": task_mean(formal, "predict_error_persistence/", "accuracy"),
        "T6_macro_f1": task_weighted(formal, "predict_error_persistence/", "macro_f1"),
        "T6_positive_recall": task_weighted(formal, "predict_error_persistence/", "positive_recall"),
        "answer_C4_pass": bool(answer.get("scientific_gate_pass")),
        "answer_C4_D_specific": float(answer.get("D_specific", -1)),
        "answer_C4_swap": float(answer.get("swap", -1)),
        "answer_C4_random": float(answer.get("random", 1)),
        "answer_C4_noop": float(answer.get("noop", 1)),
        "answer_C4_same_rule": float(answer.get("same_rule", 1)),
        "diagnosis_C4_pass": bool(diagnosis.get("pass")),
        "diagnosis_C4_D_specific": float(diagnosis.get("semantic_D_specific", -1)),
        "diagnosis_C4_swap_to_donor": float(
            ((diagnosis.get("cells") or {}).get("swap_to_donor") or {}).get(
                "semantic_equivalence_rate", -1
            )
        ),
        "diagnosis_C4_noop_retention": float(diagnosis.get("noop_recipient_retention", -1)),
        "diagnosis_C4_same_rule_retention": float(diagnosis.get("same_rule_recipient_retention", -1)),
        "diagnosis_C4_random_stable": float(diagnosis.get("random_matches_either_stable_target_rate", 1)),
    }
    floors: dict[str, Any] = {
        "T1": 0.55,
        "T2": 0.20,
        "T3_eedi_semantic": 0.20,
        "T3_synthetic_semantic": 0.45,
        "T3_schema": 0.95,
        "T5": 0.35,
        "T6": 0.35,
        "T6_macro_f1": 0.30,
        "T6_positive_recall": "strictly_greater_than_zero",
        "answer_C4_D_specific": 0.15,
        "answer_C4_random_max": 0.25,
        "answer_C4_noop_max": 0.05,
        "answer_C4_same_rule_max": 0.10,
        "diagnosis_C4_D_specific": 0.15,
        "diagnosis_C4_swap_to_donor": 0.50,
        "diagnosis_C4_noop_retention": 0.75,
        "diagnosis_C4_same_rule_retention": 0.65,
        "diagnosis_C4_random_stable_max": 0.25,
    }
    qualified = (
        checks["answer_C4_pass"]
        and checks["diagnosis_C4_pass"]
        and all(
            checks[key] >= floors[key]
            for key in (
                "T1",
                "T2",
                "T3_eedi_semantic",
                "T3_synthetic_semantic",
                "T3_schema",
                "T5",
                "T6",
                "T6_macro_f1",
                "answer_C4_D_specific",
                "diagnosis_C4_D_specific",
                "diagnosis_C4_swap_to_donor",
                "diagnosis_C4_noop_retention",
                "diagnosis_C4_same_rule_retention",
            )
        )
        and checks["T6_positive_recall"] > 0
        and checks["answer_C4_random"] <= floors["answer_C4_random_max"]
        and checks["answer_C4_noop"] <= floors["answer_C4_noop_max"]
        and checks["answer_C4_same_rule"] <= floors["answer_C4_same_rule_max"]
        and checks["diagnosis_C4_random_stable"] <= floors["diagnosis_C4_random_stable_max"]
    )
    candidate = {
        "slug": SLUG,
        "qualified": qualified,
        "checks": checks,
        "floors": floors,
        "artifact_sha256": artifacts,
        "remote_root": REMOTE_ROOT,
        "finalfit_variant": FINALFIT_VARIANT,
        "finalfit_job": FINALFIT_JOB,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
    }
    return {
        "stage": "state8_matched_cp3000_semantic_joint_promotion_v1",
        "status": "complete",
        "pass": qualified,
        "selected": candidate if qualified else None,
        "candidates": [candidate],
        "selection_scope": "unified_tasks_plus_answer_and_diagnosis_c4",
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rescue-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.summary, args.rescue_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
