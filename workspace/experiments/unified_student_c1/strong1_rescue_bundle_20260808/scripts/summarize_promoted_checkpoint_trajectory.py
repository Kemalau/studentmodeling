#!/usr/bin/env python3
"""Summarize a frozen T1/T3/T6 plus C4 promoted-checkpoint trajectory screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SUPPORT = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 25,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 25,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 25,
    "predict_correctness/dbe_kt22/temporal_warm_item_binary": 25,
    "predict_correctness/eedi_response/profile_heldout_binary": 25,
    "predict_correctness/foundationalassist/temporal_warm_item_binary": 25,
    "predict_correctness/malrule/synthetic_cross_template_binary": 25,
    "predict_correctness/malrule/synthetic_same_template_binary": 25,
    "predict_correctness/xes3g5m/temporal_warm_item_binary": 25,
    "predict_error_persistence/dbe_kt22/strict_exact_recurrence_binary": 12,
    "predict_error_persistence/foundationalassist/strict_exact_recurrence_binary": 25,
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def weighted(groups: dict[str, Any], prefix: str, metric: str) -> float:
    selected = [
        row for name, row in groups.items() if name.startswith(prefix + "/")
    ]
    if not selected or any(metric not in row or int(row.get("n", 0)) <= 0 for row in selected):
        raise ValueError(f"Incomplete metric={metric} prefix={prefix}")
    support = sum(int(row["n"]) for row in selected)
    return sum(float(row[metric]) * int(row["n"]) for row in selected) / support


def task_metrics(path: Path) -> dict[str, Any]:
    value = read(path)
    groups = value.get("metrics_by_group") or {}
    exact_support = set(groups) == set(EXPECTED_SUPPORT) and all(
        int(groups[name].get("n", -1)) == support
        for name, support in EXPECTED_SUPPORT.items()
    )
    if value.get("state_architecture") != "layerwise":
        raise ValueError("State architecture drift")
    if value.get("state_shape") != [64, 5120]:
        raise ValueError("State shape drift")
    if value.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Raw-history leak")
    if value.get("persistent_history_kv_cache") is not False:
        raise ValueError("Persistent-history KV leak")
    expected_rows = sum(EXPECTED_SUPPORT.values())
    return {
        "exact_support": exact_support,
        "sampled_rows": int(value.get("sampled_rows", -1)),
        "expected_rows": expected_rows,
        "T1_macro_f1": weighted(groups, "predict_correctness", "macro_f1"),
        "T6_macro_f1": weighted(groups, "predict_error_persistence", "macro_f1"),
        "T3_format_valid_rate": weighted(
            groups, "diagnose_misconception", "format_valid_rate"
        ),
        "T3_normalized_exact_match": weighted(
            groups, "diagnose_misconception", "normalized_exact_match"
        ),
    }


def c4_metrics(path: Path) -> dict[str, Any]:
    value = read(path)
    summary = value.get("summary") or {}
    clean = summary.get("clean") or {}
    return {
        "all_pairs_n": int(summary.get("all_pairs_n", -1)),
        "clean_n": int(summary.get("clean_n", -1)),
        "swap": float((clean.get("swap") or {}).get("flip_to_A_rate", float("nan"))),
        "random": float((clean.get("random") or {}).get("flip_to_A_rate", float("nan"))),
        "same_rule": float((clean.get("samemal") or {}).get("flip_to_A_rate", float("nan"))),
        "noop": float((clean.get("noop") or {}).get("flip_to_A_rate", float("nan"))),
        "D_specific": float(summary.get("D_specific", float("nan"))),
        "scientific_gate_pass": bool(summary.get("scientific_gate_pass")),
        "checks": summary.get("checks") or {},
    }


def summarize(
    metrics: Path,
    semantic: Path,
    c4: Path,
    *,
    checkpoint_round: int,
    baseline_metrics: Path | None = None,
    baseline_semantic: Path | None = None,
) -> dict[str, Any]:
    current = task_metrics(metrics)
    semantic_value = read(semantic)
    if int(semantic_value.get("checkpoint_round", -1)) != checkpoint_round:
        raise ValueError("Semantic checkpoint round drift")
    causal = c4_metrics(c4)
    checks = {
        "exact_protocol_support": current["exact_support"],
        "exact_sampled_rows": current["sampled_rows"] == current["expected_rows"],
        "T1_not_catastrophic": current["T1_macro_f1"] >= 0.40,
        "T6_not_catastrophic": current["T6_macro_f1"] >= 0.25,
        "T3_format_valid": current["T3_format_valid_rate"] >= 0.90,
        "T3_semantic_retained": bool(semantic_value.get("pass")),
        "C4_persists": causal["scientific_gate_pass"],
    }
    out: dict[str, Any] = {
        "stage": "promoted_checkpoint_trajectory_screen",
        "checkpoint_round": checkpoint_round,
        "diagnostic_only": True,
        "formal_final_evaluation_required": True,
        "tasks": current,
        "T3_semantic": {
            key: semantic_value.get(key)
            for key in (
                "support",
                "semantic_accuracy",
                "reference_semantic_accuracy",
                "retention_ratio",
                "minimum_retention_ratio",
                "pass",
                "judge_backend",
                "judge_rubric_sha256",
            )
        },
        "C4": causal,
        "checks": checks,
        "joint_diagnostic_pass": all(checks.values()),
    }
    if baseline_metrics is not None:
        baseline = task_metrics(baseline_metrics)
        out["round1000_task_baseline"] = baseline
        out["task_delta_from_round1000"] = {
            key: current[key] - baseline[key]
            for key in (
                "T1_macro_f1",
                "T6_macro_f1",
                "T3_format_valid_rate",
                "T3_normalized_exact_match",
            )
        }
    if baseline_semantic is not None:
        baseline = read(baseline_semantic)
        out["round1000_semantic_baseline"] = {
            key: baseline.get(key)
            for key in ("semantic_accuracy", "retention_ratio", "pass")
        }
        out["semantic_accuracy_delta_from_round1000"] = float(
            semantic_value["semantic_accuracy"]
        ) - float(baseline["semantic_accuracy"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--c4", type=Path, required=True)
    parser.add_argument("--checkpoint-round", type=int, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--baseline-semantic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(
        args.metrics,
        args.semantic,
        args.c4,
        checkpoint_round=args.checkpoint_round,
        baseline_metrics=args.baseline_metrics,
        baseline_semantic=args.baseline_semantic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["joint_diagnostic_pass"] else 2)


if __name__ == "__main__":
    main()
