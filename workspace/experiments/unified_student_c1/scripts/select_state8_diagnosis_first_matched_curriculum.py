#!/usr/bin/env python3
"""Joint promotion gate for the schema-matched Teacher-to-State8 curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from select_state8_strong1_widen8_bridge import T3_GROUPS, task_mean, task_weighted
except ImportError:
    from scripts.select_state8_strong1_widen8_bridge import T3_GROUPS, task_mean, task_weighted


SLUG = "diagnosis_first_matched_seed67"
REMOTE_ROOT = (
    "/mnt/amulet/studentmodeling/diagnostics/"
    "state8_qualified_teacher_curriculum_highkd_v1_immutable_"
    "taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4"
)
FINALFIT_VARIANT = "diagnosis_first_matched_seed67"
FINALFIT_JOB = "state8-diagnosis-first-matched-seed67-unified-finalfit-seed42"


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def one(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern}, observed={paths}")
    return paths[0]


def select(root: Path, full_teacher_gate: Path, full_teacher_provenance: Path) -> dict[str, Any]:
    formal = read(one(root / "download", "**/eval/formal_screen/metrics.json"))
    answer = read(one(root / "download", "**/eval/answer_c4_150/recurrent_causal_gate.json"))
    matched = read(one(root / "download", "**/metrics/matched_teacher_curriculum.json"))
    diagnosis_recipe = read(one(root / "download", "**/metrics/diagnosis_first_teacher_curriculum.json"))
    immutable = read(one(root / "download", "**/metrics/immutable_receipt.json"))
    t3 = read(root / "t3_semantic/metrics.json")
    diagnosis = read(root / "c4_semantic/summary.json")
    full_gate = read(full_teacher_gate)
    full_provenance = read(full_teacher_provenance)

    artifacts = formal.get("artifact_sha256") or {}
    if set(artifacts) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Formal artifact binding drift")
    if answer.get("artifact_sha256") != artifacts:
        raise ValueError("Answer-C4 artifact binding drift")
    if (diagnosis.get("generation") or {}).get("artifact_sha256") != artifacts:
        raise ValueError("Diagnosis-C4 artifact binding drift")
    if immutable.get("sha256") != artifacts:
        raise ValueError("Immutable receipt artifact binding drift")
    if formal.get("state_shape") != [64, 8, 5120]:
        raise ValueError("State shape drift")
    for value in (formal, answer, diagnosis.get("generation") or {}):
        if value.get("raw_history_visible_to_decoder") is not False:
            raise ValueError("Decoder history leakage")
        if value.get("persistent_history_kv_cache") is not False:
            raise ValueError("Persistent history KV cache")

    for recipe in (matched, diagnosis_recipe):
        if recipe.get("state_token_count") != 8:
            raise ValueError("State-token count drift")
        if recipe.get("history_dropout_schedule") != [0.0, 0.25, 0.5, 0.75, 1.0]:
            raise ValueError("History curriculum drift")
        if recipe.get("final_inference_route") != "strict_state_only":
            raise ValueError("Final inference route drift")
        if recipe.get("raw_history_visible_to_decoder") is not False:
            raise ValueError("Recipe history leakage")
    if matched.get("teacher_output_schema") != "diagnosis_evidence_label_v1":
        raise ValueError("Matched teacher schema drift")
    if matched.get("teacher_diagnosis_field_order") != ["diagnosis", "evidence", "label"]:
        raise ValueError("Matched teacher field order drift")
    if matched.get("semantic_judge_authoritative") is not True or matched.get("exact_match_authoritative") is not False:
        raise ValueError("Matched metric authority drift")
    if diagnosis_recipe.get("field_order") != ["diagnosis", "evidence", "label"]:
        raise ValueError("Diagnosis-first field order drift")
    if diagnosis_recipe.get("canonical_label_role") != "auxiliary_ordinary_weight":
        raise ValueError("Canonical-label role drift")

    if full_gate.get("pass") is not True or full_gate.get("support") != 1494:
        raise ValueError("Full Teacher gate is not passed")
    if full_gate.get("judge_backend") != "trapi" or full_gate.get("exact_match_is_authoritative") is not False:
        raise ValueError("Full Teacher authority drift")
    if full_provenance.get("status") != "complete" or full_provenance.get("teacher_gate_pass") is not True:
        raise ValueError("Full Teacher provenance gate drift")
    if full_provenance.get("evaluation_shards") != 16 or full_provenance.get("evaluation_rows") != 1494:
        raise ValueError("Full Teacher evaluation coverage drift")
    if full_provenance.get("output_schema") != "diagnosis_evidence_label_v1":
        raise ValueError("Full Teacher output schema drift")

    if t3.get("judge_backend") != "trapi" or t3.get("judge_errors") != 0:
        raise ValueError("Student T3 semantic judgment incomplete")
    groups = t3.get("metrics_by_group") or {}
    if set(groups) != set(T3_GROUPS):
        raise ValueError("Student T3 group drift")
    semantic: dict[str, float] = {}
    schema_total = 0.0
    support = 0
    for name, short in T3_GROUPS.items():
        row = groups[name]; n = int(row.get("llm_judge_n", -1))
        if n != 25 or int(row.get("llm_judge_api_or_cache", -1)) != n or int(row.get("llm_judge_exact_shortcuts", -1)) != 0:
            raise ValueError(f"Student T3 authority drift: {short}")
        semantic[short] = float(row["llm_judge_accuracy"])
        schema_total += float(row["natural_schema_valid_rate"]) * n
        support += n
    synthetic = 0.5 * (semantic["cross"] + semantic["same"])
    schema = schema_total / support

    formal_groups = formal.get("metrics_by_group") or {}
    answer_summary = answer.get("summary") or {}; answer_clean = answer_summary.get("clean") or {}
    diagnosis_cells = diagnosis.get("cells") or {}
    checks = {
        "T1": task_mean(formal_groups, "predict_correctness/", "accuracy"),
        "T2": task_mean(formal_groups, "predict_response/", "math_equivalence"),
        "T3_eedi_semantic": semantic["eedi"],
        "T3_synthetic_semantic": synthetic,
        "T3_schema": schema,
        "T5": task_mean(formal_groups, "predict_prospective_error/", "joint_math_equivalence"),
        "T6": task_mean(formal_groups, "predict_error_persistence/", "accuracy"),
        "T6_macro_f1": task_weighted(formal_groups, "predict_error_persistence/", "macro_f1"),
        "T6_positive_recall": task_weighted(formal_groups, "predict_error_persistence/", "positive_recall"),
        "answer_C4_pass": bool(answer_summary.get("scientific_gate_pass")),
        "answer_C4_D_specific": float(answer_summary.get("D_specific", -1)),
        "answer_C4_random": float((answer_clean.get("random") or {}).get("flip_to_A_rate", 1)),
        "answer_C4_noop": float((answer_clean.get("noop") or {}).get("flip_to_A_rate", 1)),
        "answer_C4_same_rule": float((answer_clean.get("samemal") or {}).get("flip_to_A_rate", 1)),
        "diagnosis_C4_pass": bool(diagnosis.get("pass")),
        "diagnosis_C4_D_specific": float(diagnosis.get("semantic_D_specific", -1)),
        "diagnosis_C4_swap_to_donor": float((diagnosis_cells.get("swap_to_donor") or {}).get("semantic_equivalence_rate", -1)),
        "diagnosis_C4_noop_retention": float(diagnosis.get("noop_recipient_retention", -1)),
        "diagnosis_C4_same_rule_retention": float(diagnosis.get("same_rule_recipient_retention", -1)),
        "diagnosis_C4_random_stable": float(diagnosis.get("random_matches_either_stable_target_rate", 1)),
    }
    floors: dict[str, Any] = {
        "T1": 0.55, "T2": 0.20, "T3_eedi_semantic": 0.20,
        "T3_synthetic_semantic": 0.45, "T3_schema": 0.95,
        "T5": 0.35, "T6": 0.35, "T6_macro_f1": 0.30,
        "T6_positive_recall": "strictly_greater_than_zero",
        "answer_C4_D_specific": 0.15, "answer_C4_random_max": 0.25,
        "answer_C4_noop_max": 0.05, "answer_C4_same_rule_max": 0.10,
        "diagnosis_C4_D_specific": 0.15, "diagnosis_C4_swap_to_donor": 0.50,
        "diagnosis_C4_noop_retention": 0.75, "diagnosis_C4_same_rule_retention": 0.65,
        "diagnosis_C4_random_stable_max": 0.25,
    }
    qualified = (
        checks["answer_C4_pass"] and checks["diagnosis_C4_pass"]
        and all(checks[key] >= floors[key] for key in (
            "T1", "T2", "T3_eedi_semantic", "T3_synthetic_semantic", "T3_schema",
            "T5", "T6", "T6_macro_f1", "answer_C4_D_specific",
            "diagnosis_C4_D_specific", "diagnosis_C4_swap_to_donor",
            "diagnosis_C4_noop_retention", "diagnosis_C4_same_rule_retention",
        ))
        and checks["T6_positive_recall"] > 0
        and checks["answer_C4_random"] <= floors["answer_C4_random_max"]
        and checks["answer_C4_noop"] <= floors["answer_C4_noop_max"]
        and checks["answer_C4_same_rule"] <= floors["answer_C4_same_rule_max"]
        and checks["diagnosis_C4_random_stable"] <= floors["diagnosis_C4_random_stable_max"]
    )
    score = (
        .10 * checks["T1"] + .08 * checks["T2"] + .12 * checks["T3_eedi_semantic"]
        + .12 * checks["T3_synthetic_semantic"] + .05 * checks["T3_schema"]
        + .08 * checks["T5"] + .05 * checks["T6_macro_f1"]
        + .15 * checks["answer_C4_D_specific"] + .15 * checks["diagnosis_C4_D_specific"]
        + .10 * checks["diagnosis_C4_swap_to_donor"] - .10 * checks["diagnosis_C4_random_stable"]
    )
    candidate = {
        "slug": SLUG, "qualified": qualified, "score": score,
        "checks": checks, "floors": floors, "artifact_sha256": artifacts,
        "remote_root": REMOTE_ROOT, "finalfit_variant": FINALFIT_VARIANT,
        "finalfit_job": FINALFIT_JOB,
        "full_teacher_gate_sha256": __import__("hashlib").sha256(full_teacher_gate.read_bytes()).hexdigest(),
        "full_teacher_provenance_sha256": __import__("hashlib").sha256(full_teacher_provenance.read_bytes()).hexdigest(),
        "semantic_judge_authoritative": True, "exact_match_authoritative": False,
    }
    return {
        "stage": "state8_diagnosis_first_matched_joint_promotion_v1",
        "pass": qualified, "selected": candidate if qualified else None,
        "candidates": [candidate],
        "selection_scope": "full_teacher_plus_unified_tasks_plus_answer_and_diagnosis_c4",
        "semantic_judge_authoritative": True, "exact_match_authoritative": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-teacher-gate", type=Path, required=True)
    parser.add_argument("--full-teacher-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.root, args.full_teacher_gate, args.full_teacher_provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
