#!/usr/bin/env python3
"""Train explicit response, mastery, prospective-error, and persistence baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


TASKS = {
    "predict_response",
    "estimate_mastery",
    "predict_prospective_error",
    "predict_error_persistence",
}


def read(path: Path, deduplicate: bool) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row.get("task") or "")
            if task not in TASKS:
                continue
            original_id = str((row.get("metadata") or {}).get("original_example_id") or row["example_id"])
            key = (task, original_id)
            if deduplicate and key in seen:
                continue
            seen.add(key)
            rows[task].append(row)
    return rows


def source(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")


def qid(row: dict[str, Any]) -> str:
    return f"{source(row)}::{(row.get('metadata') or {}).get('question_id', '')}"


def skill(row: dict[str, Any]) -> str:
    return f"{source(row)}::{(row.get('metadata') or {}).get('skill_id', '')}"


def split_train_valid(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, valid = [], []
    for row in rows:
        value = int(hashlib.sha256(str(row["student_id"]).encode("utf-8")).hexdigest()[:8], 16) % 100
        (valid if value < 10 else train).append(row)
    return train, valid


def event_outcome(event: dict[str, Any]) -> int:
    return int(event.get("correctness", event.get("is_correct", 0)))


def event_kcs(event: dict[str, Any]) -> set[str]:
    values = event.get("kc_ids") or []
    if values:
        return {str(value) for value in values}
    value = event.get("skill_id") or event.get("family")
    return {str(value)} if value else set()


def base_features(row: dict[str, Any]) -> dict[str, Any]:
    history = row.get("history") or []
    outcomes = [event_outcome(event) for event in history]
    query_kcs = {str(value) for value in ((row.get("query") or {}).get("kc_ids") or [])}
    same_kc = [event_outcome(event) for event in history if query_kcs.intersection(event_kcs(event))]
    recent = outcomes[-3:]
    return {
        "source": source(row),
        "question": qid(row),
        "skill": skill(row),
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


def classification_metrics(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) == 2 else None,
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def fit_logistic(
    train_features: list[dict[str, Any]],
    train_labels: list[Any],
    valid_features: list[dict[str, Any]],
    valid_labels: list[Any],
    binary: bool,
) -> tuple[DictVectorizer, LogisticRegression, float]:
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_features)
    x_valid = vectorizer.transform(valid_features)
    for matrix in (x_train, x_valid):
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    best = None
    for regularization in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=regularization, max_iter=500, solver="saga", random_state=42)
        model.fit(x_train, train_labels)
        probabilities = model.predict_proba(x_valid)
        score = log_loss(valid_labels, probabilities, labels=model.classes_)
        if best is None or score < best[0]:
            best = (score, model)
    assert best is not None
    return vectorizer, best[1], best[0]


def transform(vectorizer: DictVectorizer, features: list[dict[str, Any]]):
    matrix = vectorizer.transform(features)
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def train_t2(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train, valid = split_train_valid(train_rows)

    def expand(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
        features, labels = [], []
        for row in rows:
            query = row.get("query") or {}
            correct_label = str(query.get("correct_response_id") or query.get("correct_response") or query.get("correct_answer") or "")
            for candidate in row["candidate_labels"]:
                feature = base_features(row)
                feature.update(
                    {
                        "candidate": f"{qid(row)}::{candidate}",
                        "candidate_is_correct": int(str(candidate) == correct_label),
                        "correct_x_history": int(str(candidate) == correct_label) * float(feature["history_rate"]),
                    }
                )
                features.append(feature)
                labels.append(int(str(candidate) == str(row["target_label"])))
        return features, labels

    train_features, train_labels = expand(train)
    valid_features, valid_labels = expand(valid)
    vectorizer, model, valid_nll = fit_logistic(train_features, train_labels, valid_features, valid_labels, True)
    predictions = []
    for row in test_rows:
        row_features, _ = expand([row])
        scores = model.predict_proba(transform(vectorizer, row_features))[:, list(model.classes_).index(1)]
        predictions.append(str(row["candidate_labels"][int(np.argmax(scores))]))
    labels = [str(row["target_label"]) for row in test_rows]
    result = classification_metrics(labels, predictions)
    result.update({"model": "conditional response logistic", "valid_candidate_nll": valid_nll})
    return result


def train_t4(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train, valid = split_train_valid(train_rows)
    vectorizer, model, valid_nll = fit_logistic(
        [base_features(row) for row in train],
        [str(row["target_label"]) for row in train],
        [base_features(row) for row in valid],
        [str(row["target_label"]) for row in valid],
        False,
    )
    predictions = model.predict(transform(vectorizer, [base_features(row) for row in test_rows])).tolist()
    result = classification_metrics([str(row["target_label"]) for row in test_rows], predictions)
    result.update({"model": "ordinal-style PFA multinomial logistic", "valid_nll": valid_nll})
    return result


def train_t5(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train, valid = split_train_valid(train_rows)
    vectorizer, model, valid_nll = fit_logistic(
        [base_features(row) for row in train],
        [int((row.get("metadata") or {}).get("target_correctness", 0)) for row in train],
        [base_features(row) for row in valid],
        [int((row.get("metadata") or {}).get("target_correctness", 0)) for row in valid],
        True,
    )
    probabilities = model.predict_proba(transform(vectorizer, [base_features(row) for row in test_rows]))[:, list(model.classes_).index(1)]
    incorrect_modes: dict[str, Counter[str]] = defaultdict(Counter)
    global_incorrect: Counter[str] = Counter()
    for row in train:
        if int((row.get("metadata") or {}).get("target_correctness", 0)) == 0:
            incorrect_modes[qid(row)][str(row["target_label"])] += 1
            global_incorrect[str(row["target_label"])] += 1
    response_predictions = []
    for row, probability in zip(test_rows, probabilities):
        query = row.get("query") or {}
        correct_label = str(query.get("correct_response_id") or query.get("correct_response") or "")
        if probability >= 0.5 and correct_label in row["candidate_labels"]:
            prediction = correct_label
        else:
            counts = incorrect_modes.get(qid(row), Counter())
            allowed = [str(label) for label in row["candidate_labels"] if str(label) != correct_label]
            prediction = max(allowed, key=lambda label: (counts.get(label, global_incorrect.get(label, 0)), label))
        response_predictions.append(prediction)
    correctness_labels = np.asarray([int((row.get("metadata") or {}).get("target_correctness", 0)) for row in test_rows])
    result = {
        "model": "PFA-style correctness + item distractor mode",
        "valid_correctness_nll": valid_nll,
        "correctness": binary_metrics(correctness_labels, probabilities),
        "exact_response": classification_metrics([str(row["target_label"]) for row in test_rows], response_predictions),
    }
    return result


def train_t6(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train, valid = split_train_valid(train_rows)
    vectorizer, model, valid_nll = fit_logistic(
        [base_features(row) for row in train],
        [int(row["target_label"] == "persistent") for row in train],
        [base_features(row) for row in valid],
        [int(row["target_label"] == "persistent") for row in valid],
        True,
    )
    probabilities = model.predict_proba(transform(vectorizer, [base_features(row) for row in test_rows]))[:, list(model.classes_).index(1)]
    labels = np.asarray([int(row["target_label"] == "persistent") for row in test_rows])
    result = binary_metrics(labels, probabilities)
    result.update({"model": "persistence logistic survival proxy", "valid_nll": valid_nll})
    result["by_source"] = {}
    for source_name in sorted({source(row) for row in test_rows}):
        indices = [index for index, row in enumerate(test_rows) if source(row) == source_name]
        result["by_source"][source_name] = binary_metrics(labels[indices], probabilities[indices])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    train = read(args.train_jsonl, deduplicate=True)
    test = read(args.test_jsonl, deduplicate=False)
    result = {
        "data": {task: {"train": len(train[task]), "test": len(test[task])} for task in sorted(TASKS)},
        "T2": train_t2(train["predict_response"], test["predict_response"]),
        "T4": train_t4(train["estimate_mastery"], test["estimate_mastery"]),
        "T5": train_t5(train["predict_prospective_error"], test["predict_prospective_error"]),
        "T6": train_t6(train["predict_error_persistence"], test["predict_error_persistence"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
