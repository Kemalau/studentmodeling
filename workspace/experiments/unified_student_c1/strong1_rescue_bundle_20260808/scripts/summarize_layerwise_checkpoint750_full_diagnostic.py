#!/usr/bin/env python3
"""Summarize the full-support, closed-state round-750 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_layerwise_unified_final_eval import (
    FROZEN_GROUP_SUPPORT,
    FROZEN_TOTAL_ROWS,
    REFERENCE,
    weighted_metric,
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def validate_task_metrics(path: Path) -> dict[str, Any]:
    value = read(path)
    if value.get("state_architecture") != "layerwise":
        raise ValueError(f"Architecture drift: {path}")
    if value.get("state_shape") != [64, 5120]:
        raise ValueError(f"State shape drift: {path}")
    if value.get("raw_history_visible_to_decoder") is not False:
        raise ValueError(f"Raw history leak: {path}")
    if value.get("persistent_history_kv_cache") is not False:
        raise ValueError(f"History KV leak: {path}")
    return value


def summarize(
    formal_root: Path,
    t6_metrics: Path,
    c4_gate: Path,
    t3_semantic_metrics: Path | None = None,
) -> dict[str, Any]:
    formal_paths = sorted(formal_root.glob("*/metrics.json"))
    if len(formal_paths) != 4:
        raise ValueError(f"Expected four full formal metric files: {formal_paths}")
    task_rows = [validate_task_metrics(path) for path in formal_paths]
    task_rows.append(validate_task_metrics(t6_metrics))
    groups: dict[str, Any] = {}
    sampled_rows = 0
    sources = []
    for path, value in [
        *zip(formal_paths, task_rows[:4], strict=True),
        (t6_metrics, task_rows[4]),
    ]:
        overlap = set(groups).intersection(value["metrics_by_group"])
        if overlap:
            raise ValueError(f"Duplicate protocol groups in {path}: {overlap}")
        groups.update(value["metrics_by_group"])
        sampled_rows += int(value["sampled_rows"])
        sources.append({"path": str(path), "sampled_rows": int(value["sampled_rows"])})

    tasks = {name.split("/", 1)[0] for name in groups}
    required = {
        "predict_correctness",
        "predict_response",
        "diagnose_misconception",
        "predict_prospective_error",
        "predict_error_persistence",
    }
    if tasks != required:
        raise ValueError(f"Task groups drift: {tasks}")

    performance = {}
    for task, reference in REFERENCE.items():
        row = weighted_metric(groups, task, reference["metric"])
        row.update({"reference": reference["value"], "min_ratio": reference["min_ratio"]})
        row["ratio"] = row["score"] / row["reference"]
        row["pass"] = row["ratio"] >= row["min_ratio"]
        performance[task] = row

    t3_groups = [
        value for name, value in groups.items()
        if name.startswith("diagnose_misconception/")
    ]
    t3_support = sum(int(row["n"]) for row in t3_groups)
    t3_format = sum(
        float(row["format_valid_rate"]) * int(row["n"]) for row in t3_groups
    ) / t3_support
    t3_exact = sum(
        float(row["normalized_exact_match"]) * int(row["n"]) for row in t3_groups
    ) / t3_support
    semantic = None
    if t3_semantic_metrics is not None:
        semantic_value = read(t3_semantic_metrics)
        semantic_groups = semantic_value.get("metrics_by_group") or {}
        semantic_n = sum(int(row.get("llm_judge_n", 0)) for row in semantic_groups.values())
        semantic_score = (
            sum(
                float(row["llm_judge_accuracy"]) * int(row["llm_judge_n"])
                for row in semantic_groups.values()
            ) / semantic_n
            if semantic_n
            else None
        )
        semantic = {
            "path": str(t3_semantic_metrics),
            "support": semantic_n,
            "llm_judge_accuracy": semantic_score,
            "judge_backend": semantic_value.get("judge_backend"),
            "judge_effective_model": semantic_value.get("judge_effective_model"),
        }

    c4 = read(c4_gate)
    if c4.get("state_architecture") != "layerwise" or c4.get("state_shape") != [64, 5120]:
        raise ValueError("C4 architecture drift")
    if c4.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("C4 raw-history leak")
    if c4.get("persistent_history_kv_cache") is not False:
        raise ValueError("C4 history-KV leak")
    c4_summary = c4["summary"]
    clean = c4_summary["clean"]

    checks = {
        "all_protocol_groups_present": set(groups) == set(FROZEN_GROUP_SUPPORT),
        "all_protocol_support_exact": all(
            name in groups and int(groups[name].get("n", -1)) == support
            for name, support in FROZEN_GROUP_SUPPORT.items()
        ),
        "full_formal_row_support": (
            sampled_rows == FROZEN_TOTAL_ROWS
            and sum(int(row.get("n", 0)) for row in groups.values()) == FROZEN_TOTAL_ROWS
        ),
        "task_performance_retained": all(row["pass"] for row in performance.values()),
        "T3_format_valid": t3_format >= 0.90,
        "C4_support": int(c4_summary.get("clean_n", 0)) >= 50,
        "C4_specificity": float(c4_summary.get("D_specific", -1)) >= 0.15,
        "C4_noop": float(clean["noop"]["flip_to_A_rate"]) <= 0.05,
        "C4_same_rule": float(clean["samemal"]["flip_to_A_rate"]) <= 0.10,
    }
    return {
        "stage": "layerwise_checkpoint750_full_local_diagnostic",
        "checkpoint_round": 750,
        "diagnostic_only": True,
        "formal_final_evaluation_required": True,
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "state_architecture": "layerwise",
        "state_shape": [64, 5120],
        "sampled_rows": sampled_rows,
        "checks": checks,
        "diagnostic_gate_pass": all(checks.values()),
        "performance_retention": performance,
        "T3": {
            "support": t3_support,
            "format_valid_rate": t3_format,
            "normalized_exact_match": t3_exact,
            "semantic": semantic,
        },
        "C4": {
            "path": str(c4_gate),
            "clean_n": int(c4_summary["clean_n"]),
            "swap": float(clean["swap"]["flip_to_A_rate"]),
            "random": float(clean["random"]["flip_to_A_rate"]),
            "same_rule": float(clean["samemal"]["flip_to_A_rate"]),
            "noop": float(clean["noop"]["flip_to_A_rate"]),
            "D_specific": float(c4_summary["D_specific"]),
        },
        "metric_sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--t6-metrics", type=Path, required=True)
    parser.add_argument("--c4-gate", type=Path, required=True)
    parser.add_argument("--t3-semantic-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(
        args.formal_root,
        args.t6_metrics,
        args.c4_gate,
        args.t3_semantic_metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["diagnostic_gate_pass"] else 2)


if __name__ == "__main__":
    main()
