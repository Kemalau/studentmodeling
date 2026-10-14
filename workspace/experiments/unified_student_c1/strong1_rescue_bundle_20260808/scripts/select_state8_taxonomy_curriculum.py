#!/usr/bin/env python3
"""Select a five-stage curriculum using semantic T3 and unified task retention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


T3_GROUPS = (
    "diagnose_misconception/eedi_mining/open_taxonomy_generation",
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation",
    "diagnose_misconception/malrule/synthetic_same_template_open_generation",
)
REMOTE_ROOTS = {
    "taxonomy_t3heavy": "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1",
    "taxonomy_balanced": "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_balanced_curriculum_v1_immutable_taxonomy_semantic_v1",
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def task_mean(groups: dict[str, Any], prefix: str, metric: str) -> float:
    rows = [row for name, row in groups.items() if name.startswith(prefix)]
    if not rows or any(metric not in row for row in rows):
        raise ValueError(f"Missing {prefix}/{metric}")
    return sum(float(row[metric]) for row in rows) / len(rows)


def score(slug: str, root: Path) -> dict[str, Any]:
    if slug not in REMOTE_ROOTS:
        raise ValueError(f"Unknown curriculum slug: {slug}")
    semantic = read(root / "t3_semantic/metrics.json")
    formal_paths = list((root / "download").glob("**/eval/formal_screen/metrics.json"))
    receipt_paths = list((root / "download").glob("**/metrics/immutable_receipt.json"))
    taxonomy_paths = list((root / "download").glob("**/metrics/taxonomy_teacher_curriculum.json"))
    c4_path = root / "c4_semantic/summary.json"
    if len(formal_paths) != 1 or len(receipt_paths) != 1 or len(taxonomy_paths) != 1:
        raise ValueError(f"Artifact cardinality drift: {slug}")
    formal = read(formal_paths[0])
    receipt = read(receipt_paths[0])
    taxonomy = read(taxonomy_paths[0])
    c4 = read(c4_path)

    if semantic.get("judge_backend") != "trapi" or semantic.get("judge_errors") != 0:
        raise ValueError(f"Incomplete authoritative judge: {slug}")
    semantic_groups = semantic.get("metrics_by_group") or {}
    if set(semantic_groups) != set(T3_GROUPS):
        raise ValueError(f"Semantic group drift: {slug}")
    support = sum(int(semantic_groups[name]["llm_judge_n"]) for name in T3_GROUPS)
    if support <= 0 or any(
        int(semantic_groups[name].get("llm_judge_exact_shortcuts", -1)) != 0
        for name in T3_GROUPS
    ):
        raise ValueError(f"Exact-match authority or empty support: {slug}")
    t3 = sum(
        float(semantic_groups[name]["llm_judge_accuracy"])
        * int(semantic_groups[name]["llm_judge_n"])
        for name in T3_GROUPS
    ) / support
    schema = sum(
        float(semantic_groups[name]["natural_schema_valid_rate"])
        * int(semantic_groups[name]["llm_judge_n"])
        for name in T3_GROUPS
    ) / support

    groups = formal.get("metrics_by_group") or {}
    t1 = task_mean(groups, "predict_correctness/", "accuracy")
    t2 = task_mean(groups, "predict_response/", "math_equivalence")
    t5 = task_mean(groups, "predict_prospective_error/", "joint_math_equivalence")
    t6 = task_mean(groups, "predict_error_persistence/", "accuracy")
    if formal.get("state_shape") != [64, 8, 5120]:
        raise ValueError(f"State shape drift: {slug}")
    if formal.get("raw_history_visible_to_decoder") is not False:
        raise ValueError(f"History leakage: {slug}")
    if receipt.get("status") != "complete" or receipt.get("state_shape") != [64, 8, 5120]:
        raise ValueError(f"Immutable receipt drift: {slug}")
    if taxonomy.get("semantic_judge_authoritative") is not True:
        raise ValueError(f"Semantic authority drift: {slug}")
    if taxonomy.get("exact_label_metric_authoritative") is not False:
        raise ValueError(f"Exact-label authority drift: {slug}")
    if c4.get("semantic_judge_authoritative") is not True:
        raise ValueError(f"C4 lacks authoritative semantic judgment: {slug}")
    generation_artifacts = (c4.get("generation") or {}).get("artifact_sha256")
    if generation_artifacts != receipt.get("sha256"):
        raise ValueError(f"C4/curriculum artifact binding drift: {slug}")

    checks = {
        "T1": t1, "T2": t2, "T3_semantic": t3, "T3_schema": schema,
        "T5": t5, "T6": t6, "C4_pass": bool(c4.get("pass")),
        "C4_D_specific": float(c4.get("semantic_D_specific", float("-inf"))),
        "C4_same_rule_retention": float(c4.get("same_rule_recipient_retention", -1.0)),
        "C4_noop_retention": float(c4.get("noop_recipient_retention", -1.0)),
        "C4_random_stable": float(c4.get("random_matches_either_stable_target_rate", 1.0)),
    }
    floors = {"T1": 0.45, "T2": 0.15, "T3_semantic": 0.25, "T3_schema": 0.90, "T5": 0.25, "T6": 0.30}
    qualified = all(checks[name] >= floor for name, floor in floors.items())
    qualified = qualified and checks["C4_pass"]
    aggregate = (
        0.15 * t1 + 0.10 * t2 + 0.20 * t3 + 0.05 * schema
        + 0.10 * t5 + 0.05 * t6
        + 0.20 * checks["C4_D_specific"]
        + 0.10 * checks["C4_same_rule_retention"]
        + 0.05 * checks["C4_noop_retention"]
        - 0.10 * checks["C4_random_stable"]
    )
    return {
        "slug": slug,
        "qualified": qualified,
        "score": aggregate,
        "checks": checks,
        "floors": floors,
        "remote_root": REMOTE_ROOTS[slug],
        "artifact_sha256": receipt["sha256"],
        "receipt_stage": receipt["stage"],
        "formal_metrics": str(formal_paths[0]),
        "semantic_metrics": str(root / "t3_semantic/metrics.json"),
        "c4_semantic_metrics": str(c4_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("SLUG", "ROOT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [score(slug, Path(root)) for slug, root in args.candidate]
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(eligible, key=lambda row: (row["score"], row["slug"] == "taxonomy_balanced")) if eligible else None
    result = {
        "stage": "state8_taxonomy_curriculum_selection_v2",
        "pass": selected is not None,
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: row["slug"]),
        "selection_scope": "semantic_T3_plus_T1_T2_T5_T6_plus_authoritative_C4",
        "exact_match_authoritative": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if selected else 2)


if __name__ == "__main__":
    main()
