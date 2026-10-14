#!/usr/bin/env python3
"""Summarize one external-state-teacher diagnostic without weakening gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_layerwise_checkpoint250_formal_screen import summarize as summarize_formal
from validate_checkpoint_t3_semantic_result import validate as validate_t3


T6_REFERENCE_MACRO_F1 = 0.5959866676584252
T6_MIN_RATIO = 0.85
TEACHER_SHA256 = "31db2cbe6163e2a32ffe71688803d75b707b7624f79af4a6a1f4a6892138835a"


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def one(root: Path, pattern: str) -> Path:
    paths = list(root.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} under {root}: {paths}")
    return paths[0]


def weighted(groups: dict[str, Any], metric: str) -> float:
    support = sum(int(row["n"]) for row in groups.values())
    if support <= 0:
        raise ValueError("Metric support must be positive")
    return sum(float(row[metric]) * int(row["n"]) for row in groups.values()) / support


def summarize(job: Path, t3_metrics: Path) -> dict[str, Any]:
    receipt = read(one(job, "task_retention_training_receipt.json"))
    completion = read(one(job, "completion.json"))
    config = read(one(job, "run_config.json"))
    view_paths = list(job.rglob("external_state_teacher_training_view.json"))
    view_paths.extend(job.rglob("gold_semantic_state_teacher_training_view.json"))
    if len(view_paths) != 1:
        raise ValueError(f"Expected one state-teacher view receipt under {job}: {view_paths}")
    teacher_view = read(view_paths[0])
    c4 = read(job / "eval/c4/recurrent_causal_gate.json")
    t6 = read(job / "eval/t6_full/metrics.json")
    formal = summarize_formal(job / "eval/formal_screen", checkpoint_round=4250)
    t3 = validate_t3(t3_metrics, checkpoint_round=4250)

    expected_teacher = config.get("external_state_teacher_sha256")
    expected_scope = str(config.get("external_state_teacher_scope") or "")
    teacher_mode = (
        "gold_semantic"
        if expected_scope == "gold_semantic_eedi_t3_layerwise_trajectory"
        else "external_direct"
    )
    semantic_check = True
    if teacher_mode == "gold_semantic":
        semantic_check = bool(
            teacher_view.get("student_observes_gold_label") is False
            and teacher_view.get("gold_label_visible_at_inference") is False
            and config.get("gold_semantic_label_visible_at_inference") is False
            and int(teacher_view.get("gold_semantic_eedi_t3_rows", 0)) > 0
        )
    teacher_check = bool(
        expected_teacher == TEACHER_SHA256
        and teacher_view.get("teacher_adapter_sha256") == TEACHER_SHA256
        and teacher_view.get("teacher_scope") == expected_scope
        and teacher_view.get("teacher_trainable") is False
        and config.get("external_state_teacher_trainable") is False
        and int(teacher_view.get("changed_malrule_t3_rows", 0)) > 0
        and semantic_check
    )
    c4_summary = c4["summary"]
    clean = c4_summary["clean"]
    t6_groups = t6["metrics_by_group"]
    t6_macro = weighted(t6_groups, "macro_f1")
    t6_recall = weighted(t6_groups, "positive_recall")
    checks = {
        "diagnostic_only": receipt.get("diagnostic_only") is True
        and receipt.get("exact_resume") is False,
        "external_teacher_hash_and_scope": teacher_check,
        "closed_layerwise": c4.get("state_architecture") == "layerwise"
        and c4.get("raw_history_visible_to_decoder") is False
        and c4.get("persistent_history_kv_cache") is False,
        "C4": bool(c4_summary.get("scientific_gate_pass")),
        "formal_screen": bool(formal.get("diagnostic_gate_pass")),
        "T3_semantic": bool(t3.get("pass")),
        "T6_retention": t6_macro / T6_REFERENCE_MACRO_F1 >= T6_MIN_RATIO
        and t6_recall > 0.0,
    }
    return {
        "stage": "external_state_teacher_eedi_full_diagnostic_v1",
        "job": job.name,
        "teacher_mode": teacher_mode,
        "teacher_scope": expected_scope,
        "diagnostic_only": True,
        "consistency_weight": float(config["consistency_weight"]),
        "learning_rate": float(config["learning_rate"]),
        "teacher_adapter_sha256": expected_teacher,
        "checks": checks,
        "all_gates_pass": all(checks.values()),
        "C4": {
            "clean_n": int(c4_summary["clean_n"]),
            "swap": float(clean["swap"]["flip_to_A_rate"]),
            "random": float(clean["random"]["flip_to_A_rate"]),
            "same_rule": float(clean["samemal"]["flip_to_A_rate"]),
            "noop": float(clean["noop"]["flip_to_A_rate"]),
            "D_specific": float(c4_summary["D_specific"]),
        },
        "T3": t3,
        "T6": {
            "support": int(t6["sampled_rows"]),
            "macro_f1": t6_macro,
            "reference_macro_f1": T6_REFERENCE_MACRO_F1,
            "retention_ratio": t6_macro / T6_REFERENCE_MACRO_F1,
            "positive_recall": t6_recall,
        },
        "formal_screen": formal,
    }


def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2 or {row["consistency_weight"] for row in rows} != {0.25, 1.0}:
        raise ValueError("Expected the matched 0.25/1.0 external-teacher arms")
    passing = [row for row in rows if row["all_gates_pass"]]
    selected = max(
        passing,
        key=lambda row: (
            float(row["T3"]["semantic_accuracy"]),
            float(row["C4"]["D_specific"]),
            float(row["T6"]["retention_ratio"]),
        ),
        default=None,
    )
    return {
        "stage": "external_state_teacher_eedi_full_selection_v1",
        "diagnostic_only": True,
        "candidate_count": len(rows),
        "pass_count": len(passing),
        "any_all_gates_pass": bool(passing),
        "selected_candidate": selected,
        "arms": sorted(rows, key=lambda row: row["consistency_weight"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, action="append", required=True)
    parser.add_argument("--t3-metrics", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.job) != len(args.t3_metrics):
        raise ValueError("Each job requires one T3 semantic metrics file")
    value = select(
        [summarize(job, metrics) for job, metrics in zip(args.job, args.t3_metrics)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
