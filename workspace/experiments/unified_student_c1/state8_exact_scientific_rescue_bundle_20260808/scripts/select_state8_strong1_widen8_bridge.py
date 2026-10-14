#!/usr/bin/env python3
"""Joint promotion gate for strong-one-slot-to-State8 bridge candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


T3_GROUPS = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": "eedi",
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": "cross",
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": "same",
}
REQUIRED_T3_RUBRIC = "t3-semantic-equivalence-v2-substantive-schema"
TRAIN_JOBS = {
    "lr5e7_seed58": "state8-strong1-widen8-bridge-lr5e7-seed58",
    "lr1e6_seed59": "state8-strong1-widen8-bridge-lr1e6-seed59",
    "single_lr5e7_seed60": "state8-strong1-widen8-single-lr5e7-seed60",
}
EVAL_JOBS = {
    "lr5e7_seed58": "state8-strong1-widen8-eval-lr5e7-seed58",
    "lr1e6_seed59": "state8-strong1-widen8-eval-lr1e6-seed59",
    "single_lr5e7_seed60": "state8-strong1-widen8-single-eval-lr5e7-seed60",
}


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


def task_mean(groups: dict[str, Any], prefix: str, metric: str) -> float:
    rows = [row for name, row in groups.items() if name.startswith(prefix)]
    if not rows or any(metric not in row for row in rows):
        raise ValueError(f"Missing {prefix}/{metric}")
    return sum(float(row[metric]) for row in rows) / len(rows)


def task_weighted(groups: dict[str, Any], prefix: str, metric: str) -> float:
    rows = [row for name, row in groups.items() if name.startswith(prefix)]
    if not rows or any(metric not in row or "n" not in row for row in rows):
        raise ValueError(f"Missing weighted {prefix}/{metric}")
    support = sum(int(row["n"]) for row in rows)
    if support <= 0:
        raise ValueError(f"Non-positive support for {prefix}/{metric}")
    return sum(float(row[metric]) * int(row["n"]) for row in rows) / support


def score(slug: str, root: Path) -> dict[str, Any]:
    if slug not in TRAIN_JOBS:
        raise ValueError(f"Unknown bridge arm: {slug}")
    train_job = TRAIN_JOBS[slug]
    eval_job = EVAL_JOBS[slug]
    formal_path = one(
        root / "train_download", f"**/{train_job}/eval/formal_screen/metrics.json"
    )
    receipt_path = one(
        root / "train_download", f"**/{train_job}/metrics/immutable_receipt.json"
    )
    taxonomy_path = one(
        root / "train_download",
        f"**/{train_job}/metrics/taxonomy_teacher_curriculum.json",
    )
    answer_c4_path = one(
        root / "eval_download",
        f"**/{eval_job}/eval/answer_c4_seed42_n150/recurrent_causal_gate.json",
    )
    formal = read(formal_path)
    receipt = read(receipt_path)
    taxonomy = read(taxonomy_path)
    t3 = read(root / slug / "t3_semantic/metrics.json")
    diagnosis_c4 = read(root / slug / "c4_semantic/summary.json")
    answer_c4 = read(answer_c4_path)

    artifacts = receipt.get("sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "adapter_model",
        "recurrent_state",
    }:
        raise ValueError(f"Immutable artifact identity missing: {slug}")
    for name, value in (
        ("formal", formal.get("artifact_sha256")),
        ("answer_c4", answer_c4.get("artifact_sha256")),
        (
            "diagnosis_c4",
            (diagnosis_c4.get("generation") or {}).get("artifact_sha256"),
        ),
    ):
        if value != artifacts:
            raise ValueError(f"Checkpoint binding drift: {slug}/{name}")
    for name, value in (("formal", formal), ("answer_c4", answer_c4)):
        if value.get("state_shape") != [64, 8, 5120]:
            raise ValueError(f"State shape drift: {slug}/{name}")
        if value.get("raw_history_visible_to_decoder") is not False:
            raise ValueError(f"History leakage: {slug}/{name}")
        if value.get("persistent_history_kv_cache") is not False:
            raise ValueError(f"Persistent history cache: {slug}/{name}")
    if receipt.get("status") != "complete" or receipt.get("state_shape") != [64, 8, 5120]:
        raise ValueError(f"Immutable receipt drift: {slug}")
    if receipt.get("raw_history_visible_to_decoder") is not False:
        raise ValueError(f"Receipt history leakage: {slug}")
    if taxonomy.get("semantic_judge_authoritative") is not True:
        raise ValueError(f"T3 semantic authority drift: {slug}")
    if taxonomy.get("exact_label_metric_authoritative") is not False:
        raise ValueError(f"Exact-label authority drift: {slug}")
    if taxonomy.get("output_schema") != "label_diagnosis_evidence_v1":
        raise ValueError(f"Natural output schema drift: {slug}")

    if t3.get("judge_backend") != "trapi" or int(t3.get("judge_errors", -1)) != 0:
        raise ValueError(f"Incomplete authoritative T3 judgment: {slug}")
    if t3.get("judge_rubric") != REQUIRED_T3_RUBRIC:
        raise ValueError(f"Stale or non-substantive T3 rubric: {slug}")
    groups = t3.get("metrics_by_group") or {}
    if set(groups) != set(T3_GROUPS):
        raise ValueError(f"T3 group drift: {slug}")
    values: dict[str, float] = {}
    schema_numerator = 0.0
    support = 0
    for name, short in T3_GROUPS.items():
        row = groups[name]
        n = int(row.get("llm_judge_n", -1))
        if n <= 0 or int(row.get("llm_judge_exact_shortcuts", -1)) != 0:
            raise ValueError(f"T3 support/authority drift: {slug}/{short}")
        if int(row.get("llm_judge_schema_valid_n", -1)) < 0:
            raise ValueError(f"T3 substantive-schema audit missing: {slug}/{short}")
        if "llm_judge_raw_semantic_accuracy" not in row:
            raise ValueError(f"T3 raw semantic audit missing: {slug}/{short}")
        values[short] = float(row["llm_judge_accuracy"])
        schema_numerator += float(row["natural_schema_valid_rate"]) * n
        support += n
    synthetic = 0.5 * (values["cross"] + values["same"])
    schema = schema_numerator / support

    formal_groups = formal.get("metrics_by_group") or {}
    answer_summary = answer_c4.get("summary") or {}
    answer_clean = answer_summary.get("clean") or {}
    diagnosis_cells = diagnosis_c4.get("cells") or {}
    checks = {
        "T1": task_mean(formal_groups, "predict_correctness/", "accuracy"),
        "T2": task_mean(formal_groups, "predict_response/", "math_equivalence"),
        "T3_eedi_semantic": values["eedi"],
        "T3_synthetic_semantic": synthetic,
        "T3_schema": schema,
        "T5": task_mean(
            formal_groups, "predict_prospective_error/", "joint_math_equivalence"
        ),
        "T6": task_mean(formal_groups, "predict_error_persistence/", "accuracy"),
        "T6_macro_f1": task_weighted(
            formal_groups, "predict_error_persistence/", "macro_f1"
        ),
        "T6_positive_recall": task_weighted(
            formal_groups, "predict_error_persistence/", "positive_recall"
        ),
        "answer_C4_pass": bool(answer_summary.get("scientific_gate_pass")),
        "answer_C4_D_specific": float(answer_summary.get("D_specific", -1.0)),
        "answer_C4_swap": float(
            ((answer_clean.get("swap") or {}).get("flip_to_A_rate", -1.0))
        ),
        "answer_C4_random": float(
            ((answer_clean.get("random") or {}).get("flip_to_A_rate", 1.0))
        ),
        "answer_C4_same_rule": float(
            ((answer_clean.get("samemal") or {}).get("flip_to_A_rate", 1.0))
        ),
        "answer_C4_noop": float(
            ((answer_clean.get("noop") or {}).get("flip_to_A_rate", 1.0))
        ),
        "diagnosis_C4_pass": bool(diagnosis_c4.get("pass")),
        "diagnosis_C4_D_specific": float(
            diagnosis_c4.get("semantic_D_specific", -1.0)
        ),
        "diagnosis_C4_swap_to_donor": float(
            (diagnosis_cells.get("swap_to_donor") or {}).get(
                "semantic_equivalence_rate", -1.0
            )
        ),
        "diagnosis_C4_same_rule_retention": float(
            diagnosis_c4.get("same_rule_recipient_retention", -1.0)
        ),
        "diagnosis_C4_noop_retention": float(
            diagnosis_c4.get("noop_recipient_retention", -1.0)
        ),
        "diagnosis_C4_random_stable": float(
            diagnosis_c4.get("random_matches_either_stable_target_rate", 1.0)
        ),
    }
    floors = {
        "T1": 0.55,
        "T2": 0.20,
        "T3_eedi_semantic": 0.17,
        "T3_synthetic_semantic": 0.45,
        "T3_schema": 0.95,
        "T5": 0.35,
        "T6": 0.35,
        "T6_macro_f1": 0.30,
        "T6_positive_recall": "strictly_greater_than_zero",
        "answer_C4_D_specific": 0.15,
        "answer_C4_same_rule_max": 0.10,
        "answer_C4_noop_max": 0.05,
        "diagnosis_C4_D_specific": 0.15,
        "diagnosis_C4_swap_to_donor": 0.50,
        "diagnosis_C4_same_rule_retention": 0.65,
        "diagnosis_C4_noop_retention": 0.75,
        "diagnosis_C4_random_stable_max": 0.25,
    }
    qualified = (
        checks["answer_C4_pass"]
        and checks["diagnosis_C4_pass"]
        and checks["T1"] >= floors["T1"]
        and checks["T2"] >= floors["T2"]
        and checks["T3_eedi_semantic"] >= floors["T3_eedi_semantic"]
        and checks["T3_synthetic_semantic"] >= floors["T3_synthetic_semantic"]
        and checks["T3_schema"] >= floors["T3_schema"]
        and checks["T5"] >= floors["T5"]
        and checks["T6"] >= floors["T6"]
        and checks["T6_macro_f1"] >= floors["T6_macro_f1"]
        and checks["T6_positive_recall"] > 0.0
        and checks["answer_C4_D_specific"] >= floors["answer_C4_D_specific"]
        and checks["answer_C4_same_rule"] <= floors["answer_C4_same_rule_max"]
        and checks["answer_C4_noop"] <= floors["answer_C4_noop_max"]
        and checks["diagnosis_C4_D_specific"]
        >= floors["diagnosis_C4_D_specific"]
        and checks["diagnosis_C4_swap_to_donor"]
        >= floors["diagnosis_C4_swap_to_donor"]
        and checks["diagnosis_C4_same_rule_retention"]
        >= floors["diagnosis_C4_same_rule_retention"]
        and checks["diagnosis_C4_noop_retention"]
        >= floors["diagnosis_C4_noop_retention"]
        and checks["diagnosis_C4_random_stable"]
        <= floors["diagnosis_C4_random_stable_max"]
    )
    aggregate = (
        0.10 * checks["T1"]
        + 0.08 * checks["T2"]
        + 0.12 * checks["T3_eedi_semantic"]
        + 0.12 * checks["T3_synthetic_semantic"]
        + 0.05 * checks["T3_schema"]
        + 0.08 * checks["T5"]
        + 0.05 * checks["T6_macro_f1"]
        + 0.15 * checks["answer_C4_D_specific"]
        + 0.15 * checks["diagnosis_C4_D_specific"]
        + 0.10 * checks["diagnosis_C4_swap_to_donor"]
        - 0.10 * checks["diagnosis_C4_random_stable"]
    )
    return {
        "slug": slug,
        "qualified": qualified,
        "score": aggregate,
        "checks": checks,
        "floors": floors,
        "artifact_sha256": artifacts,
        "remote_root": receipt.get("remote_root"),
        "exact_match_authoritative": False,
        "semantic_judge_authoritative": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", choices=sorted(TRAIN_JOBS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    slugs = args.candidate or sorted(TRAIN_JOBS)
    candidates = [score(slug, args.root) for slug in slugs]
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(eligible, key=lambda row: row["score"]) if eligible else None
    result = {
        "stage": "state8_strong1_widen8_joint_promotion_v1",
        "pass": selected is not None,
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: row["slug"]),
        "selection_scope": (
            "strict_state_only_T1_T2_T3_T5_T6_plus_answer_C4_plus_"
            "authoritative_natural_diagnosis_C4"
        ),
        "exact_match_authoritative": False,
        "semantic_judge_authoritative": True,
        "multi_seed_replication_pending": selected is not None,
        "full_259845_fit_pending": selected is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if selected is not None else 2)


if __name__ == "__main__":
    main()
