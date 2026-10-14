#!/usr/bin/env python3
"""Train Rasch-style IRT and PFA baselines for canonical T1 rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def read(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("task") == "predict_correctness":
                    rows.append(row)
    return rows


def source(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")


def question(row: dict[str, Any]) -> str:
    return f"{source(row)}::{(row.get('metadata') or {}).get('question_id', '')}"


def skill(row: dict[str, Any]) -> str:
    return f"{source(row)}::{(row.get('metadata') or {}).get('skill_id', '')}"


def target(row: dict[str, Any]) -> int:
    return int(row["target_label"] == "correct")


def metrics(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([target(row) for row in rows])
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "n": len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) == 2 else None,
    }
    by_source = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[source(row)].append(index)
    for name, indices in sorted(groups.items()):
        source_rows = [rows[index] for index in indices]
        by_source[name] = metrics_without_groups(source_rows, probabilities[indices])
    result["by_source"] = by_source
    return result


def metrics_without_groups(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([target(row) for row in rows])
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) == 2 else None,
    }


def irt_features(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [[str(row["student_id"]), question(row)] for row in rows]


def fit_irt(train: list[dict[str, Any]], valid: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    y_train = np.asarray([target(row) for row in train])
    y_valid = np.asarray([target(row) for row in valid])
    for regularization in (0.1, 1.0, 10.0):
        pipeline = Pipeline(
            [
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                (
                    "logistic",
                    LogisticRegression(C=regularization, max_iter=500, solver="saga", random_state=42),
                ),
            ]
        )
        pipeline.fit(irt_features(train), y_train)
        valid_prob = pipeline.predict_proba(irt_features(valid))[:, 1]
        valid_nll = log_loss(y_valid, valid_prob, labels=[0, 1])
        if best is None or valid_nll < best[0]:
            best = (valid_nll, regularization, pipeline)
    assert best is not None
    test_prob = best[2].predict_proba(irt_features(test))[:, 1]
    return {"selected_c": best[1], "valid_nll": best[0], "test": metrics(test, test_prob)}


def event_kcs(event: dict[str, Any]) -> set[str]:
    values = event.get("kc_ids") or []
    if values:
        return {str(value) for value in values}
    value = event.get("skill_id") or event.get("family")
    return {str(value)} if value else set()


def pfa_counts(row: dict[str, Any]) -> tuple[int, int]:
    query_kcs = {str(value) for value in ((row.get("query") or {}).get("kc_ids") or [])}
    successes = failures = 0
    for event in row.get("history") or []:
        if query_kcs and not query_kcs.intersection(event_kcs(event)):
            continue
        outcome = int(event.get("correctness", event.get("is_correct", 0)))
        successes += outcome
        failures += 1 - outcome
    return successes, failures


def pfa_matrix(rows: list[dict[str, Any]], skill_map: dict[str, int]) -> csr_matrix:
    row_indices = []
    column_indices = []
    values = []
    for row_index, row in enumerate(rows):
        skill_index = skill_map.get(skill(row))
        if skill_index is None:
            continue
        successes, failures = pfa_counts(row)
        for offset, value in enumerate((1.0, float(successes), float(failures))):
            row_indices.append(row_index)
            column_indices.append(skill_index * 3 + offset)
            values.append(value)
    return csr_matrix((values, (row_indices, column_indices)), shape=(len(rows), len(skill_map) * 3))


def fit_pfa(train: list[dict[str, Any]], valid: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    skill_map = {value: index for index, value in enumerate(sorted({skill(row) for row in train}))}
    x_train = pfa_matrix(train, skill_map)
    x_valid = pfa_matrix(valid, skill_map)
    x_test = pfa_matrix(test, skill_map)
    y_train = np.asarray([target(row) for row in train])
    y_valid = np.asarray([target(row) for row in valid])
    best = None
    for regularization in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=regularization, max_iter=500, solver="saga", random_state=42)
        model.fit(x_train, y_train)
        valid_prob = model.predict_proba(x_valid)[:, 1]
        valid_nll = log_loss(y_valid, valid_prob, labels=[0, 1])
        if best is None or valid_nll < best[0]:
            best = (valid_nll, regularization, model)
    assert best is not None
    test_prob = best[2].predict_proba(x_test)[:, 1]
    return {
        "selected_c": best[1],
        "valid_nll": best[0],
        "n_skills": len(skill_map),
        "test": metrics(test, test_prob),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    train, valid, test = read(args.train_jsonl), read(args.valid_jsonl), read(args.test_jsonl)
    result = {
        "data": {"train": len(train), "valid": len(valid), "test": len(test)},
        "rasch_irt": fit_irt(train, valid, test),
        "pfa": fit_pfa(train, valid, test),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
