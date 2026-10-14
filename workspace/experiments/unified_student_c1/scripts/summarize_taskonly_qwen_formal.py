#!/usr/bin/env python3
"""Recompute and summarize the five task-only Qwen formal baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_formal_protocols import binary_metrics, generation_metrics


TASKS = {
    "T1": ("predict_correctness", 10_913),
    "T2": ("predict_response", 3_693),
    "T3": ("diagnose_misconception", 1_494),
    "T4": ("predict_prospective_error", 4_600),
    "T5": ("predict_error_persistence", 976),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def weighted(groups: dict[str, Any], key: str) -> tuple[float, int]:
    rows = [(float(value[key]), int(value["n"])) for value in groups.values() if key in value]
    if not rows:
        raise ValueError(f"No semantic groups contain {key}")
    total = sum(count for _, count in rows)
    return sum(value * count for value, count in rows) / total, total


def summarize(prediction_files: list[Path], semantic_metrics: Path) -> dict[str, Any]:
    by_task: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for path in prediction_files:
        records = read_jsonl(path)
        observed = {str(row.get("task")) for row in records}
        if len(observed) != 1:
            raise ValueError(f"Prediction task cardinality drift: {path}: {observed}")
        task = observed.pop()
        if task in by_task:
            raise ValueError(f"Duplicate prediction file for {task}")
        by_task[task] = (path, records)

    expected_tasks = {value[0] for value in TASKS.values()}
    if set(by_task) != expected_tasks:
        raise ValueError(f"Task coverage drift: {set(by_task)} != {expected_tasks}")

    semantic = read_json(semantic_metrics)
    if semantic.get("judge_backend") != "trapi":
        raise ValueError("T3 semantic result is not backed by TRAPI")
    if int(semantic.get("generation_rows", -1)) != TASKS["T3"][1]:
        raise ValueError("T3 semantic support drift")
    if int(semantic.get("judge_errors", -1)) != 0:
        raise ValueError("T3 semantic judge has errors")

    cells: dict[str, Any] = {}
    adapter_hashes: dict[str, str] = {}
    for public, (task, expected) in TASKS.items():
        path, records = by_task[task]
        ids = [str(row["example_id"]) for row in records]
        if len(records) != expected or len(ids) != len(set(ids)):
            raise ValueError(f"{public} support/duplicate drift: {len(records)}")
        if public == "T1":
            value = binary_metrics(records, "correct")
            cell = {"primary_name": "accuracy", "primary": value["accuracy"], "n": value["n"], "auroc": value["auroc"]}
        elif public == "T2":
            value = generation_metrics(records, task)
            cell = {"primary_name": "math_equivalence", "primary": value["math_equivalence"], "n": value["n"], "format_valid_rate": value["format_valid_rate"]}
        elif public == "T3":
            score, count = weighted(semantic["metrics_by_group"], "llm_judge_accuracy")
            if count != expected:
                raise ValueError(f"T3 judged support drift: {count}")
            cell = {"primary_name": "semantic_accuracy", "primary": score, "n": count, "judge_backend": "trapi", "judge_identity": semantic.get("judge_identity")}
        elif public == "T4":
            value = generation_metrics(records, task)
            cell = {"primary_name": "joint_math_equivalence", "primary": value["joint_math_equivalence"], "n": value["n"], "correctness_accuracy": value["correctness_accuracy"], "format_valid_rate": value["format_valid_rate"]}
        else:
            value = binary_metrics(records, "persistent")
            cell = {"primary_name": "accuracy", "primary": value["accuracy"], "n": value["n"], "auroc": value["auroc"]}
        metrics = read_json(path.parent / "metrics.json")
        adapter_hashes[public] = str(metrics["artifact_sha256"]["adapter_model"])
        cells[public] = cell

    return {
        "stage": "taskonly_qwen_formal_summary_v1",
        "status": "complete",
        "model": "Qwen3-32B task-only QLoRA (one epoch)",
        "same_backbone_as_unified": True,
        "test_used_for_training": False,
        "cells": cells,
        "adapter_sha256": adapter_hashes,
    }


def markdown(value: dict[str, Any]) -> str:
    cells = value["cells"]
    return "\n".join(
        [
            "# Task-only Qwen3-32B formal baselines",
            "",
            "| Model | T1 Acc / AUROC | T2 MathEq | T3 Semantic | T4 JointMath / CorrAcc | T5 Acc / AUROC |",
            "|---|---:|---:|---:|---:|---:|",
            (
                "| Qwen3-32B task-only QLoRA | "
                f"{cells['T1']['primary']:.3f} / {cells['T1']['auroc']:.3f} | "
                f"{cells['T2']['primary']:.3f} | {cells['T3']['primary']:.3f} | "
                f"{cells['T4']['primary']:.3f} / {cells['T4']['correctness_accuracy']:.3f} | "
                f"{cells['T5']['primary']:.3f} / {cells['T5']['auroc']:.3f} |"
            ),
            "",
            "T3 uses the independent TRAPI semantic judge; exact label match is not the primary metric.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--t3-semantic-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.predictions, args.t3_semantic_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown(value), encoding="utf-8")
    print(markdown(value))


if __name__ == "__main__":
    main()
