#!/usr/bin/env python3
"""Train the appendix-only T6 future-KC-performance baseline.

Public T6 is the legacy ``estimate_mastery`` task.  Its target is an observed
future same-KC performance band, not latent mastery.  This script deliberately
uses only history/query identifiers and past outcomes as features; the future
outcomes used to construct the target are audited but never exposed to the
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
)


LABELS = ("low", "medium", "high")
ORDINAL = {label: index for index, label in enumerate(LABELS)}


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("task") != "estimate_mastery":
                    raise ValueError(f"Unexpected task in {path}: {row.get('task')}")
                if row.get("target_label") not in LABELS:
                    raise ValueError(f"Unexpected T6 label: {row.get('target_label')}")
                rows.append(row)
    return rows


def source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("source") or row.get("dataset") or "unknown")


def prefixed_student(row: dict[str, Any]) -> str:
    return f"{source(row)}::{row['student_id']}"


def event_outcome(event: dict[str, Any]) -> int:
    return int(event.get("correctness", event.get("is_correct", 0)))


def event_kcs(event: dict[str, Any]) -> set[str]:
    values = event.get("kc_ids") or []
    return {str(value) for value in values}


def features(row: dict[str, Any]) -> dict[str, Any]:
    history = row.get("history") or []
    outcomes = [event_outcome(event) for event in history]
    query = row.get("query") or {}
    query_kcs = {str(value) for value in (query.get("kc_ids") or [])}
    same_kc = [
        event_outcome(event)
        for event in history
        if query_kcs.intersection(event_kcs(event))
    ]
    recent = outcomes[-3:]
    metadata = row.get("metadata") or {}
    return {
        "source": source(row),
        "question": f"{source(row)}::{metadata.get('question_id', query.get('question_id', ''))}",
        "skill": f"{source(row)}::{metadata.get('skill_id', '')}",
        "history_len": len(outcomes),
        "history_success": sum(outcomes),
        "history_failure": len(outcomes) - sum(outcomes),
        "history_rate": sum(outcomes) / len(outcomes) if outcomes else 0.5,
        "same_kc_len": len(same_kc),
        "same_kc_success": sum(same_kc),
        "same_kc_failure": len(same_kc) - sum(same_kc),
        "same_kc_rate": sum(same_kc) / len(same_kc) if same_kc else 0.5,
        "recent_rate": sum(recent) / len(recent) if recent else 0.5,
        "last_correct": outcomes[-1] if outcomes else 0.5,
    }


def digest(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def metric_block(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    gold_ord = np.asarray([ORDINAL[label] for label in labels])
    pred_ord = np.asarray([ORDINAL[label] for label in predictions])
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "ordinal_mae": float(np.abs(gold_ord - pred_ord).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(gold_ord, pred_ord, weights="quadratic")),
        "gold_distribution": dict(sorted(Counter(labels).items())),
        "prediction_distribution": dict(sorted(Counter(predictions).items())),
    }


def source_majority(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, Counter[str]] = {}
    for row in train:
        counts.setdefault(source(row), Counter())[str(row["target_label"])] += 1
    return [counts[source(row)].most_common(1)[0][0] for row in test]


def history_band(row: dict[str, Any]) -> str:
    feature = features(row)
    rate = float(feature["same_kc_rate"] if feature["same_kc_len"] else feature["history_rate"])
    if rate < 0.4:
        return "low"
    if rate < 0.7:
        return "medium"
    return "high"


def audit_splits(
    train: list[dict[str, Any]], valid: list[dict[str, Any]], test: list[dict[str, Any]]
) -> dict[str, Any]:
    students = [{prefixed_student(row) for row in split} for split in (train, valid, test)]
    ids = [{str(row["example_id"]) for row in split} for split in (train, valid, test)]
    future_counts = Counter(
        len((row.get("metadata") or {}).get("future_same_kc_outcomes") or [])
        for row in train + valid + test
    )
    checks = {
        "train_valid_student_overlap": len(students[0] & students[1]),
        "train_test_student_overlap": len(students[0] & students[2]),
        "valid_test_student_overlap": len(students[1] & students[2]),
        "train_valid_example_overlap": len(ids[0] & ids[1]),
        "train_test_example_overlap": len(ids[0] & ids[2]),
        "valid_test_example_overlap": len(ids[1] & ids[2]),
    }
    if any(checks.values()):
        raise ValueError(f"Split leakage detected: {checks}")
    return {
        "split_leakage_checks": checks,
        "future_opportunity_count_distribution": {
            str(key): value for key, value in sorted(future_counts.items())
        },
        "future_outcomes_used_as_features": False,
    }


def train_and_evaluate(
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    class_weight: str | None = None,
    selection_metric: str = "nll",
) -> tuple[dict[str, Any], list[str]]:
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([features(row) for row in train])
    x_valid = vectorizer.transform([features(row) for row in valid])
    for matrix in (x_train, x_valid):
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    y_train = [str(row["target_label"]) for row in train]
    y_valid = [str(row["target_label"]) for row in valid]
    candidates = []
    for regularization in (0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=regularization,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
            class_weight=class_weight,
        )
        model.fit(x_train, y_train)
        valid_predictions = [str(value) for value in model.predict(x_valid)]
        validation = {
            "nll": float(log_loss(y_valid, model.predict_proba(x_valid), labels=model.classes_)),
            "macro_f1": float(
                f1_score(y_valid, valid_predictions, labels=LABELS, average="macro", zero_division=0)
            ),
        }
        candidates.append((regularization, validation))
    if selection_metric == "macro_f1":
        best_c, selected_validation = max(
            candidates, key=lambda item: (item[1]["macro_f1"], -item[1]["nll"], -item[0])
        )
    elif selection_metric == "nll":
        best_c, selected_validation = min(
            candidates, key=lambda item: (item[1]["nll"], -item[1]["macro_f1"], item[0])
        )
    else:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")

    fit_rows = train + valid
    final_vectorizer = DictVectorizer(sparse=True)
    x_fit = final_vectorizer.fit_transform([features(row) for row in fit_rows])
    x_test = final_vectorizer.transform([features(row) for row in test])
    for matrix in (x_fit, x_test):
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    model = LogisticRegression(
        C=best_c,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
        class_weight=class_weight,
    )
    model.fit(x_fit, [str(row["target_label"]) for row in fit_rows])
    predictions = [str(value) for value in model.predict(x_test)]
    labels = [str(row["target_label"]) for row in test]
    overall = metric_block(labels, predictions)
    overall["nll"] = float(log_loss(labels, model.predict_proba(x_test), labels=model.classes_))
    by_source = {}
    for source_name in sorted({source(row) for row in test}):
        indices = [index for index, row in enumerate(test) if source(row) == source_name]
        by_source[source_name] = metric_block(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
    return {
        "model": "ordinal-style PFA multinomial logistic",
        "class_weight": class_weight,
        "selection_metric": selection_metric,
        "selected_c": best_c,
        "selected_validation": selected_validation,
        "validation_candidates": {
            str(regularization): validation for regularization, validation in candidates
        },
        "overall": overall,
        "by_source": by_source,
    }, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--valid-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--test-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    args = parser.parse_args()
    train = read_rows(args.train_jsonl)
    valid = read_rows(args.valid_jsonl)
    test = read_rows(args.test_jsonl)
    audit = audit_splits(train, valid, test)
    labels = [str(row["target_label"]) for row in test]
    fit_rows = train + valid
    majority_predictions = source_majority(fit_rows, test)
    history_predictions = [history_band(row) for row in test]
    ordinary_result, ordinary_predictions = train_and_evaluate(train, valid, test)
    balanced_result, balanced_predictions = train_and_evaluate(
        train,
        valid,
        test,
        class_weight="balanced",
        selection_metric="macro_f1",
    )
    value = {
        "stage": "public_t6_future_kc_baseline_v1",
        "status": "complete",
        "task": {
            "public_label": "T6",
            "internal_key": "estimate_mastery",
            "construct": "future-KC-performance band",
            "not_latent_mastery": True,
            "paper_role": "appendix_only",
        },
        "data": {"train": len(train), "valid": len(valid), "test": len(test)},
        "input_sha256": digest(args.train_jsonl + args.valid_jsonl + args.test_jsonl),
        "audit": audit,
        "baselines": {
            "source_majority": metric_block(labels, majority_predictions),
            "same_kc_history_band": metric_block(labels, history_predictions),
            "ordinal_pfa_multinomial": ordinary_result,
            "class_balanced_pfa_multinomial": balanced_result,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.predictions_jsonl is not None:
        args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_jsonl.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(test):
                record = {
                    "example_id": str(row["example_id"]),
                    "student_id": prefixed_student(row),
                    "source": source(row),
                    "gold": labels[index],
                    "predictions": {
                        "source_majority": majority_predictions[index],
                        "same_kc_history_band": history_predictions[index],
                        "ordinal_pfa_multinomial": ordinary_predictions[index],
                        "class_balanced_pfa_multinomial": balanced_predictions[index],
                    },
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
