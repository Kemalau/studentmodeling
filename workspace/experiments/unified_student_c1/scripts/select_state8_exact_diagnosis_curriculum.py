#!/usr/bin/env python3
"""Joint promotion gate for the full-history-teacher to State8 diagnosis curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from select_state8_strong1_widen8_bridge import (
        REQUIRED_T3_RUBRIC,
        T3_GROUPS,
        task_mean,
        task_weighted,
    )
except ImportError:  # package-style unit tests
    from scripts.select_state8_strong1_widen8_bridge import (
        REQUIRED_T3_RUBRIC,
        T3_GROUPS,
        task_mean,
        task_weighted,
    )


JOB = "state8-exact-diagnosis-first-curriculum-c4-seed64"
SLUG = "exact_diagnosis_curriculum_seed65"
STABLE_ROOT = (
    "/mnt/amulet/studentmodeling/diagnostics/"
    "state8_stateonly_repair_seed65_trajectory01_"
    "exact_teacher_curriculum_c4_seed65_v1"
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def one(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} below {root}, observed={paths}")
    return paths[0]


def score_candidate(
    root: Path,
    *,
    job: str,
    slug: str,
    stable_root: str,
    finalfit_variant: str,
    finalfit_job: str,
    allowed_repair_entrypoints: tuple[str, ...] = (
        "scripts/train_state8_diagnosis_first_causal_repair.py",
        "scripts/train_state8_diagnosis_first_causal_repair_exact_resume.py",
    ),
) -> dict[str, Any]:
    artifact = one(root / "download", f"**/{job}").resolve()
    formal = read(artifact / "eval/formal_t3_300/metrics.json")
    publish = read(artifact / "metrics/publish_receipt.json")
    answer = read(artifact / "eval/answer_c4_150/recurrent_causal_gate.json")
    curriculum = read(artifact / "metrics/immutable_receipt.json")
    teacher_recipe = read(artifact / "metrics/diagnosis_first_teacher_curriculum.json")
    repair_recipe = read(artifact / "metrics/diagnosis_first_causal_repair.json")
    t3 = read(root / "t3_semantic/metrics.json")
    diagnosis = read(root / "c4_semantic/summary.json")

    artifacts = publish.get("artifact_sha256") or {}
    for name, value in (
        ("formal", formal.get("artifact_sha256")),
        ("answer", answer.get("artifact_sha256")),
        ("diagnosis", (diagnosis.get("generation") or {}).get("artifact_sha256")),
    ):
        if value != artifacts:
            raise ValueError(f"Artifact binding drift: {name}")
    if publish.get("stage") != "state8_stateonly_repair_pilot_publish_v1":
        raise ValueError("Publish stage drift")
    if publish.get("train_entrypoint") not in allowed_repair_entrypoints:
        raise ValueError("Diagnosis-first repair entrypoint drift")
    if curriculum.get("status") != "complete" or curriculum.get("state_shape") != [64, 8, 5120]:
        raise ValueError("Completed State8 curriculum receipt drift")
    if curriculum.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Curriculum decoder history leakage")
    if curriculum.get("history_dropout_counts") != {
        "0.0": 1000, "0.25": 1000, "0.5": 1000, "0.75": 1000, "1.0": 1000,
    }:
        raise ValueError("Five-stage history dropout drift")
    for name, recipe in (("teacher", teacher_recipe), ("repair", repair_recipe)):
        if recipe.get("field_order") != ["diagnosis", "evidence", "label"]:
            raise ValueError(f"Diagnosis-first field order drift: {name}")
        if recipe.get("canonical_label_role") != "auxiliary_ordinary_weight":
            raise ValueError(f"Canonical label role drift: {name}")
        if float(recipe.get("label_auxiliary_weight", -1)) != 1.0:
            raise ValueError(f"Canonical label weight drift: {name}")
    if teacher_recipe.get("history_dropout_schedule") != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("Teacher curriculum schedule drift")

    for name, value in (("formal", formal), ("answer", answer)):
        if value.get("state_shape") != [64, 8, 5120]:
            raise ValueError(f"State shape drift: {name}")
        if value.get("raw_history_visible_to_decoder") is not False:
            raise ValueError(f"History leakage: {name}")
        if value.get("persistent_history_kv_cache") is not False:
            raise ValueError(f"Persistent history cache: {name}")

    if t3.get("judge_backend") != "trapi" or int(t3.get("judge_errors", -1)) != 0:
        raise ValueError("Incomplete authoritative T3 judgment")
    if t3.get("judge_rubric") != REQUIRED_T3_RUBRIC:
        raise ValueError("Stale T3 rubric")
    groups = t3.get("metrics_by_group") or {}
    if set(groups) != set(T3_GROUPS):
        raise ValueError("T3 group drift")
    semantic: dict[str, float] = {}
    schema_numerator = 0.0
    support = 0
    for name, short in T3_GROUPS.items():
        row = groups[name]
        n = int(row.get("llm_judge_n", -1))
        if n <= 0 or int(row.get("llm_judge_exact_shortcuts", -1)) != 0:
            raise ValueError(f"T3 authority drift: {short}")
        semantic[short] = float(row["llm_judge_accuracy"])
        schema_numerator += float(row["natural_schema_valid_rate"]) * n
        support += n
    synthetic = 0.5 * (semantic["cross"] + semantic["same"])
    schema = schema_numerator / support

    formal_groups = formal.get("metrics_by_group") or {}
    answer_summary = answer.get("summary") or {}
    answer_clean = answer_summary.get("clean") or {}
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
        "answer_C4_D_specific": float(answer_summary.get("D_specific", -1.0)),
        "answer_C4_same_rule": float((answer_clean.get("samemal") or {}).get("flip_to_A_rate", 1.0)),
        "answer_C4_noop": float((answer_clean.get("noop") or {}).get("flip_to_A_rate", 1.0)),
        "diagnosis_C4_pass": bool(diagnosis.get("pass")),
        "diagnosis_C4_D_specific": float(diagnosis.get("semantic_D_specific", -1.0)),
        "diagnosis_C4_swap_to_donor": float((diagnosis_cells.get("swap_to_donor") or {}).get("semantic_equivalence_rate", -1.0)),
        "diagnosis_C4_same_rule_retention": float(diagnosis.get("same_rule_recipient_retention", -1.0)),
        "diagnosis_C4_noop_retention": float(diagnosis.get("noop_recipient_retention", -1.0)),
        "diagnosis_C4_random_stable": float(diagnosis.get("random_matches_either_stable_target_rate", 1.0)),
    }
    floors = {
        "T1": 0.55, "T2": 0.20, "T3_eedi_semantic": 0.17,
        "T3_synthetic_semantic": 0.45, "T3_schema": 0.95,
        "T5": 0.35, "T6": 0.35, "T6_macro_f1": 0.30,
        "T6_positive_recall": "strictly_greater_than_zero",
        "answer_C4_D_specific": 0.15, "answer_C4_same_rule_max": 0.10,
        "answer_C4_noop_max": 0.05, "diagnosis_C4_D_specific": 0.15,
        "diagnosis_C4_swap_to_donor": 0.50,
        "diagnosis_C4_same_rule_retention": 0.65,
        "diagnosis_C4_noop_retention": 0.75,
        "diagnosis_C4_random_stable_max": 0.25,
    }
    qualified = (
        checks["answer_C4_pass"] and checks["diagnosis_C4_pass"]
        and all(checks[key] >= floors[key] for key in (
            "T1", "T2", "T3_eedi_semantic", "T3_synthetic_semantic", "T3_schema",
            "T5", "T6", "T6_macro_f1", "answer_C4_D_specific",
            "diagnosis_C4_D_specific", "diagnosis_C4_swap_to_donor",
            "diagnosis_C4_same_rule_retention", "diagnosis_C4_noop_retention",
        ))
        and checks["T6_positive_recall"] > 0.0
        and checks["answer_C4_same_rule"] <= floors["answer_C4_same_rule_max"]
        and checks["answer_C4_noop"] <= floors["answer_C4_noop_max"]
        and checks["diagnosis_C4_random_stable"] <= floors["diagnosis_C4_random_stable_max"]
    )
    aggregate = (
        .10 * checks["T1"] + .08 * checks["T2"]
        + .12 * checks["T3_eedi_semantic"] + .12 * checks["T3_synthetic_semantic"]
        + .05 * checks["T3_schema"] + .08 * checks["T5"]
        + .05 * checks["T6_macro_f1"] + .15 * checks["answer_C4_D_specific"]
        + .15 * checks["diagnosis_C4_D_specific"]
        + .10 * checks["diagnosis_C4_swap_to_donor"]
        - .10 * checks["diagnosis_C4_random_stable"]
    )
    return {
        "slug": slug, "qualified": qualified, "score": aggregate,
        "checks": checks, "floors": floors, "artifact_sha256": artifacts,
        "remote_root": stable_root,
        "finalfit_variant": finalfit_variant,
        "finalfit_job": finalfit_job,
        "curriculum_provenance": curriculum,
        "exact_match_authoritative": False, "semantic_judge_authoritative": True,
    }


def score(root: Path) -> dict[str, Any]:
    return score_candidate(
        root,
        job=JOB,
        slug=SLUG,
        stable_root=STABLE_ROOT,
        finalfit_variant="strong1_exact_diagnosis_curriculum_seed65",
        finalfit_job=(
            "state8-strong1-exact-diagnosis-curriculum-seed65-"
            "unified-finalfit-seed42"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = score(args.root)
    result = {
        "stage": "state8_exact_diagnosis_curriculum_joint_promotion_v1",
        "pass": bool(candidate["qualified"]),
        "selected": candidate if candidate["qualified"] else None,
        "candidates": [candidate],
        "selection_scope": "strict_state_only_unified_plus_answer_and_natural_diagnosis_C4",
        "exact_match_authoritative": False,
        "semantic_judge_authoritative": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if candidate["qualified"] else 2)


if __name__ == "__main__":
    main()
