#!/usr/bin/env python3
"""Select a task-anchor screen with unified retention and semantic T3/C4."""

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
ARMS = {
    "base": (
        "task_anchor_base",
        "state8-task-anchor-base-unified-finalfit-seed42",
    ),
    "c4strong": (
        "task_anchor_c4strong",
        "state8-task-anchor-c4strong-unified-finalfit-seed42",
    ),
}
SCREEN_JOBS = {
    "base": "state8-task-anchor-base-seed56",
    "c4strong": "state8-task-anchor-c4strong-seed57",
}
T6_BALANCED_SCREEN_MACRO_F1_FLOOR = 0.30


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def one(root: Path, slug: str, suffix: str) -> Path:
    # The watcher stores semantic post-processing below each arm, while the
    # immutable AMLT JSON artifacts are preserved once in the shared download
    # tree.  Also accept an arm-local download tree for standalone tests.
    paths = list((root / "download").glob(f"**/{SCREEN_JOBS[slug]}/{suffix}"))
    if not paths:
        paths = list(
            (root.parent / "download").glob(
                f"**/{SCREEN_JOBS[slug]}/{suffix}"
            )
        )
    if len(paths) != 1:
        raise ValueError(f"Expected one {suffix} for {slug} below {root}: {paths}")
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
    if slug not in ARMS:
        raise ValueError(f"Unknown task-anchor arm: {slug}")
    formal_path = one(root, slug, "eval/formal_t3_300/metrics.json")
    publish_path = one(root, slug, "metrics/publish_receipt.json")
    anchor_path = one(root, slug, "metrics/task_anchored_causal_repair.json")
    taxonomy_path = one(root, slug, "metrics/taxonomy_causal_repair.json")
    formal = read(formal_path)
    publish = read(publish_path)
    anchor = read(anchor_path)
    taxonomy = read(taxonomy_path)
    semantic = read(root / "t3_semantic/metrics.json")
    c4 = read(root / "c4_semantic/summary.json")
    if semantic.get("judge_backend") != "trapi" or int(semantic.get("judge_errors", -1)) != 0:
        raise ValueError(f"Incomplete authoritative T3 judge: {slug}")
    semantic_groups = semantic.get("metrics_by_group") or {}
    if set(semantic_groups) != set(T3_GROUPS):
        raise ValueError(f"T3 group drift: {slug}")
    if any(
        int(semantic_groups[name].get("llm_judge_exact_shortcuts", -1)) != 0
        for name in T3_GROUPS
    ):
        raise ValueError(f"Exact shortcut is forbidden: {slug}")
    support = sum(int(semantic_groups[name]["llm_judge_n"]) for name in T3_GROUPS)
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
        "T3_semantic": t3,
        "T3_schema": schema,
        "T5": task_mean(
            formal_groups, "predict_prospective_error/", "joint_math_equivalence"
        ),
        "T6": task_mean(formal_groups, "predict_error_persistence/", "accuracy"),
        "T6_macro_f1": t6_macro_f1,
        "T6_positive_recall": t6_positive_recall,
        "C4_pass": bool(c4.get("pass")),
        "C4_D_specific": float(c4.get("semantic_D_specific", -1.0)),
        "C4_same_rule_retention": float(
            c4.get("same_rule_recipient_retention", -1.0)
        ),
        "C4_noop_retention": float(c4.get("noop_recipient_retention", -1.0)),
        "C4_random_stable": float(
            c4.get("random_matches_either_stable_target_rate", 1.0)
        ),
    }
    if formal.get("state_shape") != [64, 8, 5120]:
        raise ValueError(f"State shape drift: {slug}")
    if formal.get("raw_history_visible_to_decoder") is not False:
        raise ValueError(f"Closed-state contract drift: {slug}")
    if publish.get("stage") != "state8_stateonly_repair_pilot_publish_v1":
        raise ValueError(f"Publish stage drift: {slug}")
    if publish.get("train_entrypoint") != "scripts/train_state8_task_anchored_causal_repair.py":
        raise ValueError(f"Task-anchor entrypoint drift: {slug}")
    if anchor.get("stage") != "state8_task_anchored_taxonomy_causal_repair_v1":
        raise ValueError(f"Task-anchor receipt drift: {slug}")
    if anchor.get("raw_history_visible_to_decoder") is not False:
        raise ValueError(f"Task-anchor history drift: {slug}")
    if taxonomy.get("contract") != "source_aware_no_answer_labels_v1":
        raise ValueError(f"Taxonomy contract drift: {slug}")
    if c4.get("semantic_judge_authoritative") is not True:
        raise ValueError(f"C4 authority drift: {slug}")
    if (c4.get("generation") or {}).get("artifact_sha256") != publish.get(
        "artifact_sha256"
    ):
        raise ValueError(f"C4 artifact drift: {slug}")
    floors = {
        "T1": 0.45,
        "T2": 0.15,
        "T3_semantic": 0.25,
        "T3_schema": 0.90,
        "T5": 0.25,
        "T6": 0.30,
    }
    qualified = (
        bool(checks["C4_pass"])
        and all(float(checks[name]) >= floor for name, floor in floors.items())
        # This bounded panel is label-stratified, so it cannot support a
        # retention ratio against the natural full-support T6 reference.
        # Gate collapse here and leave strict retention to final evaluation.
        and float(checks["T6_macro_f1"])
        >= T6_BALANCED_SCREEN_MACRO_F1_FLOOR
        and float(checks["T6_positive_recall"]) > 0.0
    )
    value = (
        0.15 * checks["T1"]
        + 0.10 * checks["T2"]
        + 0.20 * checks["T3_semantic"]
        + 0.05 * checks["T3_schema"]
        + 0.10 * checks["T5"]
        + 0.05 * checks["T6_macro_f1"]
        + 0.20 * checks["C4_D_specific"]
        + 0.10 * checks["C4_same_rule_retention"]
        + 0.05 * checks["C4_noop_retention"]
        - 0.10 * checks["C4_random_stable"]
    )
    return {
        "slug": slug,
        "qualified": qualified,
        "score": value,
        "checks": checks,
        "floors": {
            **floors,
            "T6_macro_f1": T6_BALANCED_SCREEN_MACRO_F1_FLOOR,
            "T6_positive_recall": "strictly_greater_than_zero",
        },
        "screen_variant": ARMS[slug][0] + "_v1",
        "finalfit_variant": ARMS[slug][0],
        "finalfit_job": ARMS[slug][1],
        "artifact_sha256": publish["artifact_sha256"],
        "exact_match_authoritative": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", nargs=2, action="append", metavar=("SLUG", "ROOT"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [score(slug, Path(root)) for slug, root in args.candidate]
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(eligible, key=lambda row: row["score"]) if eligible else None
    result = {
        "stage": "state8_task_anchor_selection_v1",
        "pass": selected is not None,
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: row["slug"]),
        "selection_scope": "strict_state_only_unified_retention_plus_semantic_T3_C4",
        "exact_match_authoritative": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if selected else 2)


if __name__ == "__main__":
    main()
