#!/usr/bin/env python3
"""Select a taxonomy consolidation arm using unified retention and semantic C4."""

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
VARIANTS = {
    "taxonomy_t3heavy": (
        "taxonomy_curriculum_t3heavy",
        "state8-taxonomy-curriculum-t3heavy-unified-finalfit-seed42",
    ),
    "taxonomy_balanced": (
        "taxonomy_curriculum_balanced",
        "state8-taxonomy-curriculum-balanced-unified-finalfit-seed42",
    ),
    "taxonomy_t3heavy_diag": (
        "taxonomy_curriculum_t3heavy_diag",
        "state8-taxonomy-curriculum-t3heavy-diag-unified-finalfit-seed42",
    ),
    "taxonomy_balanced_diag": (
        "taxonomy_curriculum_balanced_diag",
        "state8-taxonomy-curriculum-balanced-diag-unified-finalfit-seed42",
    ),
    "task_anchor_base": (
        "task_anchor_base",
        "state8-task-anchor-base-unified-finalfit-seed42",
    ),
    "task_anchor_c4strong": (
        "task_anchor_c4strong",
        "state8-task-anchor-c4strong-unified-finalfit-seed42",
    ),
}
TASK_SCREEN_JOBS = {
    "task_anchor_base": "state8-task-anchor-base-seed56",
    "task_anchor_c4strong": "state8-task-anchor-c4strong-seed57",
}
T6_BALANCED_SCREEN_MACRO_F1_FLOOR = 0.30


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


def task_weighted(groups: dict[str, Any], prefix: str, metric: str) -> float:
    rows = [row for name, row in groups.items() if name.startswith(prefix)]
    if not rows or any(metric not in row or "n" not in row for row in rows):
        raise ValueError(f"Missing weighted {prefix}/{metric}")
    support = sum(int(row["n"]) for row in rows)
    if support <= 0:
        raise ValueError(f"Non-positive support for {prefix}/{metric}")
    return sum(float(row[metric]) * int(row["n"]) for row in rows) / support


def score(slug: str, root: Path) -> dict[str, Any]:
    if slug not in VARIANTS:
        raise ValueError(f"Unknown arm: {slug}")
    if slug in TASK_SCREEN_JOBS:
        download = root.parent / "download"
        job = TASK_SCREEN_JOBS[slug]
        formal_paths = list(download.glob(f"**/{job}/eval/formal_t3_300/metrics.json"))
        publish_paths = list(download.glob(f"**/{job}/metrics/publish_receipt.json"))
        taxonomy_paths = list(download.glob(f"**/{job}/metrics/taxonomy_causal_repair.json"))
        anchor_paths = list(download.glob(f"**/{job}/metrics/task_anchored_causal_repair.json"))
    else:
        formal_paths = list((root / "download").glob("**/eval/formal_t3_300/metrics.json"))
        publish_paths = list((root / "download").glob("**/metrics/publish_receipt.json"))
        taxonomy_paths = list((root / "download").glob("**/metrics/taxonomy_causal_repair.json"))
        anchor_paths = []
    if len(formal_paths) != 1 or len(publish_paths) != 1 or len(taxonomy_paths) != 1:
        raise ValueError(f"Artifact cardinality drift: {slug}")
    formal = read(formal_paths[0]); publish = read(publish_paths[0]); taxonomy = read(taxonomy_paths[0])
    semantic = read(root / "t3_semantic/metrics.json")
    c4 = read(root / "c4_semantic/summary.json")
    if semantic.get("judge_backend") != "trapi" or int(semantic.get("judge_errors", -1)) != 0:
        raise ValueError(f"Incomplete authoritative T3 judge: {slug}")
    groups = semantic.get("metrics_by_group") or {}
    if set(groups) != set(T3_GROUPS):
        raise ValueError(f"T3 group drift: {slug}")
    if any(int(groups[name].get("llm_judge_exact_shortcuts", -1)) != 0 for name in T3_GROUPS):
        raise ValueError(f"Exact-match shortcut is forbidden: {slug}")
    support = sum(int(groups[name]["llm_judge_n"]) for name in T3_GROUPS)
    t3 = sum(float(groups[name]["llm_judge_accuracy"]) * int(groups[name]["llm_judge_n"]) for name in T3_GROUPS) / support
    schema = sum(float(groups[name]["natural_schema_valid_rate"]) * int(groups[name]["llm_judge_n"]) for name in T3_GROUPS) / support
    formal_groups = formal.get("metrics_by_group") or {}
    t6_macro_f1 = task_weighted(
        formal_groups, "predict_error_persistence/", "macro_f1"
    )
    t6_positive_recall = task_weighted(
        formal_groups, "predict_error_persistence/", "positive_recall"
    )
    checks: dict[str, Any] = {
        "T1": task_mean(formal_groups, "predict_correctness/", "accuracy"),
        "T2": task_mean(formal_groups, "predict_response/", "math_equivalence"),
        "T3_semantic": t3, "T3_schema": schema,
        "T5": task_mean(formal_groups, "predict_prospective_error/", "joint_math_equivalence"),
        "T6": task_mean(formal_groups, "predict_error_persistence/", "accuracy"),
        "T6_macro_f1": t6_macro_f1,
        "T6_positive_recall": t6_positive_recall,
        "C4_pass": bool(c4.get("pass")),
        "C4_D_specific": float(c4.get("semantic_D_specific", -1.0)),
        "C4_same_rule_retention": float(c4.get("same_rule_recipient_retention", -1.0)),
        "C4_noop_retention": float(c4.get("noop_recipient_retention", -1.0)),
        "C4_random_stable": float(c4.get("random_matches_either_stable_target_rate", 1.0)),
    }
    if formal.get("state_shape") != [64, 8, 5120] or formal.get("raw_history_visible_to_decoder") is not False:
        raise ValueError(f"Closed-state formal contract drift: {slug}")
    if publish.get("stage") != "state8_stateonly_repair_pilot_publish_v1" or publish.get("status") != "complete":
        raise ValueError(f"Publish receipt drift: {slug}")
    if publish.get("state_shape") != [64, 8, 5120] or publish.get("raw_history_visible_to_decoder") is not False:
        raise ValueError(f"Publish state contract drift: {slug}")
    if publish.get("train_entrypoint") != "scripts/train_state8_taxonomy_causal_repair.py":
        if slug not in TASK_SCREEN_JOBS or publish.get("train_entrypoint") != "scripts/train_state8_task_anchored_causal_repair.py":
            raise ValueError(f"Taxonomy/task-anchor entrypoint drift: {slug}")
    if slug in TASK_SCREEN_JOBS:
        if len(anchor_paths) != 1:
            raise ValueError(f"Task-anchor receipt cardinality drift: {slug}")
        anchor = read(anchor_paths[0])
        if anchor.get("stage") != "state8_task_anchored_taxonomy_causal_repair_v1":
            raise ValueError(f"Task-anchor receipt drift: {slug}")
        if anchor.get("raw_history_visible_to_decoder") is not False:
            raise ValueError(f"Task-anchor history drift: {slug}")
    if taxonomy.get("contract") != "source_aware_no_answer_labels_v1" or taxonomy.get("candidate_taxonomy_visible_to_decoder") is not False:
        raise ValueError(f"Taxonomy contract drift: {slug}")
    if c4.get("semantic_judge_authoritative") is not True or (c4.get("generation") or {}).get("artifact_sha256") != publish.get("artifact_sha256"):
        raise ValueError(f"C4 artifact/authority drift: {slug}")
    floors = {"T1": 0.45, "T2": 0.15, "T3_semantic": 0.25, "T3_schema": 0.90, "T5": 0.25, "T6": 0.30}
    qualified = (
        bool(checks["C4_pass"])
        and all(float(checks[name]) >= floor for name, floor in floors.items())
        # The bounded binary screen is deterministically label-stratified and
        # is not distribution-matched to the 976-row natural T6 panel.  Use it
        # only as a non-collapse gate; full-reference retention is enforced by
        # validate_state8_final_eval.py after the exact epoch.
        and float(checks["T6_macro_f1"])
        >= T6_BALANCED_SCREEN_MACRO_F1_FLOOR
        and float(checks["T6_positive_recall"]) > 0.0
    )
    score_value = (
        0.15 * checks["T1"] + 0.10 * checks["T2"] + 0.20 * checks["T3_semantic"]
        + 0.05 * checks["T3_schema"] + 0.10 * checks["T5"]
        + 0.05 * checks["T6_macro_f1"]
        + 0.20 * checks["C4_D_specific"] + 0.10 * checks["C4_same_rule_retention"]
        + 0.05 * checks["C4_noop_retention"] - 0.10 * checks["C4_random_stable"]
    )
    variant, job = VARIANTS[slug]
    return {
        "slug": slug, "qualified": qualified, "score": score_value, "checks": checks,
        "floors": {
            **floors,
            "T6_macro_f1": T6_BALANCED_SCREEN_MACRO_F1_FLOOR,
            "T6_positive_recall": "strictly_greater_than_zero",
        }, "finalfit_variant": variant, "finalfit_job": job,
        "artifact_sha256": publish["artifact_sha256"],
        "formal_metrics": str(formal_paths[0]), "semantic_metrics": str(root / "t3_semantic/metrics.json"),
        "c4_semantic_metrics": str(root / "c4_semantic/summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("SLUG", "ROOT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [score(slug, Path(root)) for slug, root in args.candidate]
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(eligible, key=lambda row: (row["score"], "balanced" in row["slug"])) if eligible else None
    result = {
        "stage": "state8_unified_candidate_selection_v1", "pass": selected is not None,
        "selected": selected, "candidates": sorted(candidates, key=lambda row: row["slug"]),
        "selection_scope": "strict_state_only_unified_retention_plus_semantic_T3_C4",
        "exact_match_authoritative": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if selected else 2)


if __name__ == "__main__":
    main()
