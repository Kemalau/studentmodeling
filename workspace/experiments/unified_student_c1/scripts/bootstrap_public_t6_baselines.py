#!/usr/bin/env python3
"""Student-cluster bootstrap intervals for appendix public-T6 baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LABELS = ("low", "medium", "high")
ORDINAL = {label: index for index, label in enumerate(LABELS)}
METRICS = ("macro_f1", "balanced_accuracy", "ordinal_mae", "quadratic_weighted_kappa")


def read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encoded_metrics(gold_ord: np.ndarray, pred_ord: np.ndarray) -> dict[str, float]:
    classes = len(LABELS)
    matrix = np.bincount(
        gold_ord * classes + pred_ord, minlength=classes * classes
    ).reshape(classes, classes)
    true_support = matrix.sum(axis=1).astype(float)
    predicted_support = matrix.sum(axis=0).astype(float)
    true_positive = np.diag(matrix).astype(float)
    f1_denominator = 2 * true_positive + (predicted_support - true_positive) + (
        true_support - true_positive
    )
    per_class_f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.zeros(classes, dtype=float),
        where=f1_denominator > 0,
    )
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros(classes, dtype=float),
        where=true_support > 0,
    )
    weight = np.fromfunction(
        lambda i, j: ((i - j) / (classes - 1)) ** 2,
        (classes, classes),
        dtype=float,
    )
    observed = float((weight * matrix).sum())
    total = float(matrix.sum())
    expected_matrix = np.outer(true_support, predicted_support) / total
    expected = float((weight * expected_matrix).sum())
    if expected == 0.0:
        qwk = 1.0 if observed == 0.0 else 0.0
    else:
        qwk = 1.0 - observed / expected
    return {
        "macro_f1": float(per_class_f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "ordinal_mae": float(np.abs(gold_ord - pred_ord).mean()),
        "quadratic_weighted_kappa": qwk,
    }


def metrics(gold: list[str], prediction: list[str]) -> dict[str, float]:
    return encoded_metrics(
        np.asarray([ORDINAL[value] for value in gold], dtype=np.int8),
        np.asarray([ORDINAL[value] for value in prediction], dtype=np.int8),
    )


def percentile(values: list[float]) -> list[float]:
    return [float(value) for value in np.quantile(np.asarray(values), [0.025, 0.975])]


def analyze(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    model_names = sorted(rows[0]["predictions"])
    gold_encoded = np.asarray([ORDINAL[str(row["gold"])] for row in rows], dtype=np.int8)
    predictions_encoded = {
        model: np.asarray(
            [ORDINAL[str(row["predictions"][model])] for row in rows], dtype=np.int8
        )
        for model in model_names
    }
    by_source_student: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        by_source_student[str(row["source"])][str(row["student_id"])].append(index)
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in METRICS} for model in model_names
    }
    for _ in range(replicates):
        indices: list[int] = []
        # Preserve each source's student-cluster count while respecting repeated
        # examples from one student. Repeated sampled clusters repeat all rows.
        for students in by_source_student.values():
            student_ids = sorted(students)
            selected = rng.choice(student_ids, size=len(student_ids), replace=True)
            for student_id in selected:
                indices.extend(students[str(student_id)])
        selected_indices = np.asarray(indices, dtype=np.int32)
        gold = gold_encoded[selected_indices]
        for model in model_names:
            value = encoded_metrics(gold, predictions_encoded[model][selected_indices])
            for metric in METRICS:
                draws[model][metric].append(value[metric])
    estimates = {}
    for model in model_names:
        point = encoded_metrics(gold_encoded, predictions_encoded[model])
        estimates[model] = {
            metric: {"value": point[metric], "ci95": percentile(draws[model][metric])}
            for metric in METRICS
        }
    reference = "source_majority"
    paired = {}
    for model in model_names:
        if model == reference:
            continue
        paired[model] = {}
        for metric in METRICS:
            differences = [
                value - reference_value
                for value, reference_value in zip(draws[model][metric], draws[reference][metric])
            ]
            direction = -1.0 if metric == "ordinal_mae" else 1.0
            paired[model][metric] = {
                "difference_vs_source_majority": (
                    estimates[model][metric]["value"] - estimates[reference][metric]["value"]
                ),
                "ci95": percentile(differences),
                "bootstrap_probability_better": float(
                    np.mean(direction * np.asarray(differences) > 0)
                ),
            }
    return {
        "stage": "public_t6_student_cluster_bootstrap_v1",
        "status": "complete",
        "rows": len(rows),
        "students": len({row["student_id"] for row in rows}),
        "sources": {
            source: {"students": len(students), "rows": sum(map(len, students.values()))}
            for source, students in sorted(by_source_student.items())
        },
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": "student cluster stratified by source",
        "estimates": estimates,
        "paired_vs_source_majority": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    value = analyze(read(args.predictions_jsonl), args.replicates, args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
