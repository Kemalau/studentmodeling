#!/usr/bin/env python3
"""Evaluate explicit T2, T5, and T6 specialists on the formal manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from formal_metric_utils import binary_metrics, extract_tag, generation_metrics


TASKS = {"predict_response", "predict_prospective_error", "predict_error_persistence"}


def read(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") in TASKS:
                rows[str(row["task"])].append(row)
    return rows


def source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("eval_source") or metadata.get("mix_source") or row.get("dataset") or "unknown")


def qid(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    value = metadata.get("question_id") or (row.get("query") or {}).get("question_id") or (row.get("query") or {}).get("surface_id") or ""
    return f"{source(row)}::{value}"


def skill(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    value = metadata.get("skill_id") or metadata.get("family") or ""
    return f"{source(row)}::{value}"


def event_outcome(event: dict[str, Any]) -> int:
    for key in ("correctness", "first_response_correctness", "is_correct"):
        if event.get(key) is not None:
            return int(event[key])
    return 0


def event_kcs(event: dict[str, Any]) -> set[str]:
    values = event.get("kc_ids") or []
    if values:
        return {str(value) for value in values}
    value = event.get("skill_id") or event.get("family")
    return {str(value)} if value else set()


def event_response(event: dict[str, Any]) -> str:
    for key in ("student_response_id", "student_response", "student_answer"):
        if event.get(key) not in (None, ""):
            return str(event[key])
    return ""


def event_correct_response(event: dict[str, Any]) -> str:
    for key in ("correct_response_id", "correct_response", "correct_answer"):
        if event.get(key) not in (None, ""):
            return str(event[key])
    return ""


def base_features(row: dict[str, Any]) -> dict[str, Any]:
    history = row.get("history") or []
    outcomes = [event_outcome(event) for event in history]
    query = row.get("query") or {}
    query_kcs = {str(value) for value in (query.get("kc_ids") or [])}
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


def transform(vectorizer: DictVectorizer, features: list[dict[str, Any]]):
    matrix = vectorizer.transform(features)
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def fit_logistic(
    train_features: list[dict[str, Any]],
    train_labels: list[int],
    valid_features: list[dict[str, Any]],
    valid_labels: list[int],
) -> tuple[DictVectorizer, LogisticRegression, float]:
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_features)
    x_valid = transform(vectorizer, valid_features)
    x_train.indices = x_train.indices.astype(np.int32, copy=False)
    x_train.indptr = x_train.indptr.astype(np.int32, copy=False)
    best = None
    for regularization in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=regularization, max_iter=500, solver="saga", random_state=42)
        model.fit(x_train, train_labels)
        probabilities = model.predict_proba(x_valid)[:, list(model.classes_).index(1)]
        score = log_loss(valid_labels, probabilities, labels=[0, 1])
        if best is None or score < best[0]:
            best = (score, model)
    assert best is not None
    return vectorizer, best[1], float(best[0])


def candidate_features(row: dict[str, Any], candidate: str) -> dict[str, Any]:
    features = base_features(row)
    responses = [event_response(event) for event in row.get("history") or []]
    correct_responses = [event_correct_response(event) for event in row.get("history") or []]
    features.update(
        {
            "candidate": f"{source(row)}::{candidate}",
            "question_candidate": f"{qid(row)}::{candidate}",
            "skill_candidate": f"{skill(row)}::{candidate}",
            "candidate_history_count": responses.count(candidate),
            "candidate_history_rate": responses.count(candidate) / len(responses) if responses else 0.0,
            "candidate_correct_count": correct_responses.count(candidate),
        }
    )
    return features


class CandidateResponseModel:
    def __init__(self, train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]]):
        def expand(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
            features, labels = [], []
            for row in rows:
                for candidate in row.get("candidate_labels") or []:
                    candidate = str(candidate)
                    features.append(candidate_features(row, candidate))
                    labels.append(int(candidate == str(row["target_label"])))
            return features, labels

        train_features, train_labels = expand(train_rows)
        valid_features, valid_labels = expand(valid_rows)
        self.available = bool(train_features and valid_features and len(set(train_labels)) == 2)
        self.valid_nll = None
        self.vectorizer = None
        self.model = None
        if self.available:
            self.vectorizer, self.model, self.valid_nll = fit_logistic(
                train_features, train_labels, valid_features, valid_labels
            )

    def predict(self, row: dict[str, Any]) -> str | None:
        labels = [str(label) for label in row.get("candidate_labels") or []]
        if not self.available or not labels or self.vectorizer is None or self.model is None:
            return None
        features = [candidate_features(row, label) for label in labels]
        probabilities = self.model.predict_proba(transform(self.vectorizer, features))[:, list(self.model.classes_).index(1)]
        return labels[int(np.argmax(probabilities))]


class ResponseModeModel:
    def __init__(self, rows: list[dict[str, Any]], conditioned: bool):
        self.conditioned = conditioned
        self.student_item: dict[tuple[str, str, int | None], Counter[str]] = defaultdict(Counter)
        self.item: dict[tuple[str, int | None], Counter[str]] = defaultdict(Counter)
        self.skill: dict[tuple[str, int | None], Counter[str]] = defaultdict(Counter)
        self.source: dict[tuple[str, int | None], Counter[str]] = defaultdict(Counter)
        for row in rows:
            label = str(row.get("target_label") or "")
            condition = target_correctness(row) if conditioned else None
            self.student_item[(str(row.get("student_id") or ""), qid(row), condition)][label] += 1
            self.item[(qid(row), condition)][label] += 1
            self.skill[(skill(row), condition)][label] += 1
            self.source[(source(row), condition)][label] += 1

    @staticmethod
    def mode(counter: Counter[str]) -> str | None:
        if not counter:
            return None
        return max(counter, key=lambda value: (counter[value], value))

    def predict(self, row: dict[str, Any], condition: int | None = None) -> str:
        keys = [
            self.student_item.get((str(row.get("student_id") or ""), qid(row), condition), Counter()),
            self.item.get((qid(row), condition), Counter()),
            self.skill.get((skill(row), condition), Counter()),
            self.source.get((source(row), condition), Counter()),
        ]
        for counter in keys:
            value = self.mode(counter)
            if value is not None:
                return value
        labels = [str(value) for value in row.get("candidate_labels") or []]
        return labels[0] if labels else ""


def clean_numeric(value: str) -> Fraction | None:
    text = str(value or "").strip().replace("$$", "").replace("$", "").replace(",", "")
    text = text.replace("−", "-").replace("÷", "/")
    if not re.fullmatch(r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+\-]?(?:\d+(?:\.\d+)?|\.\d+))?", text):
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def fraction_text(value: Fraction) -> str:
    value = value.limit_denominator(10000)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def malrule_transfer(row: dict[str, Any]) -> str | None:
    query_correct = clean_numeric(str((row.get("query") or {}).get("correct_answer") or ""))
    pairs = []
    for event in row.get("history") or []:
        correct = clean_numeric(str(event.get("correct_answer") or ""))
        response = clean_numeric(str(event.get("student_answer") or ""))
        if correct is not None and response is not None:
            pairs.append((correct, response))
    if query_correct is None or not pairs:
        return None

    transforms: list[tuple[str, Any]] = [("identity", lambda value: value)]
    offsets = [response - correct for correct, response in pairs]
    offset = median(offsets)
    transforms.append(("offset", lambda value, offset=offset: value + offset))
    ratios = [response / correct for correct, response in pairs if correct != 0]
    if ratios:
        ratio = median(ratios)
        transforms.append(("ratio", lambda value, ratio=ratio: value * ratio))
    transforms.extend(
        [
            ("negate", lambda value: -value),
            ("one_minus", lambda value: 1 - value),
            ("reciprocal", lambda value: 1 / value if value != 0 else value),
        ]
    )
    best = None
    for order, (_, transform) in enumerate(transforms):
        errors = [abs(float(transform(correct) - response)) for correct, response in pairs]
        score = (sum(errors) / len(errors), order)
        if best is None or score < best[0]:
            best = (score, transform)
    assert best is not None
    return fraction_text(best[1](query_correct))


def response_text(row: dict[str, Any], label: str) -> str:
    target = (row.get("candidate_targets") or {}).get(label)
    parsed = extract_tag(str(target or ""), "predicted_response")
    if parsed is not None:
        return parsed
    options = (row.get("query") or {}).get("options") or {}
    if label in options:
        return str(options[label])
    return label


def target_correctness(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") or {}
    if "target_correctness" in metadata:
        return int(metadata["target_correctness"])
    return int(str(extract_tag(str(row.get("target") or ""), "correctness") or "").casefold() == "correct")


def grouped_generation(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["source"], record["protocol"], str(record.get("delta") or ""))].append(record)
    result = {}
    for (source_name, protocol, delta), group in sorted(groups.items()):
        name = "/".join(part for part in (source_name, protocol, f"delta={delta}" if delta else "") if part)
        result[name] = generation_metrics(group, task)
    result["overall"] = generation_metrics(records, task)
    return result


def train_t2(
    train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_model = CandidateResponseModel(train_rows, valid_rows)
    mode_model = ResponseModeModel(train_rows + valid_rows, conditioned=False)
    records = []
    methods = Counter()
    for row in test_rows:
        label = candidate_model.predict(row)
        method = "conditional_candidate_logit"
        if label is None and source(row) == "malrule":
            label = malrule_transfer(row)
            method = "systematic_error_transfer"
        if label is None:
            label = mode_model.predict(row)
            method = "student_item_response_mode"
        methods[method] += 1
        prediction = response_text(row, str(label))
        metadata = row.get("metadata") or {}
        records.append(
            {
                "source": source(row),
                "protocol": str(metadata.get("eval_protocol") or "unknown"),
                "delta": metadata.get("prospective_delta"),
                "target": str(row["target"]),
                "raw_output": f"<predicted_response>{prediction}</predicted_response>",
            }
        )
    return {
        "model": "conditional candidate logit + response mode + systematic-error transfer",
        "valid_candidate_nll": candidate_model.valid_nll,
        "prediction_methods": dict(methods),
        "metrics": grouped_generation(records, "predict_response"),
    }


def train_t5(
    train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    vectorizer, model, valid_nll = fit_logistic(
        [base_features(row) for row in train_rows],
        [target_correctness(row) for row in train_rows],
        [base_features(row) for row in valid_rows],
        [target_correctness(row) for row in valid_rows],
    )
    probabilities = model.predict_proba(transform(vectorizer, [base_features(row) for row in test_rows]))[:, list(model.classes_).index(1)]
    mode_model = ResponseModeModel(train_rows + valid_rows, conditioned=True)
    records = []
    for row, probability in zip(test_rows, probabilities):
        correctness = int(probability >= 0.5)
        label = mode_model.predict(row, correctness)
        prediction = response_text(row, label)
        metadata = row.get("metadata") or {}
        records.append(
            {
                "source": source(row),
                "protocol": str(metadata.get("eval_protocol") or "unknown"),
                "delta": metadata.get("prospective_delta"),
                "target": str(row["target"]),
                "raw_output": (
                    f"<correctness>{'correct' if correctness else 'incorrect'}</correctness>"
                    f"<predicted_response>{prediction}</predicted_response>"
                ),
            }
        )
    return {
        "model": "PFA-style correctness logistic + conditioned response mode",
        "valid_correctness_nll": valid_nll,
        "metrics": grouped_generation(records, "predict_prospective_error"),
    }


def train_t6(
    train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    vectorizer, model, valid_nll = fit_logistic(
        [base_features(row) for row in train_rows],
        [int(str(row["target_label"]) == "persistent") for row in train_rows],
        [base_features(row) for row in valid_rows],
        [int(str(row["target_label"]) == "persistent") for row in valid_rows],
    )
    probabilities = model.predict_proba(transform(vectorizer, [base_features(row) for row in test_rows]))[:, list(model.classes_).index(1)]
    records = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row, probability in zip(test_rows, probabilities):
        record = {
            "gold": str(row["target_label"]),
            "prediction": "persistent" if probability >= 0.5 else "transient",
            "probabilities": {"persistent": float(probability), "transient": float(1 - probability)},
        }
        metadata = row.get("metadata") or {}
        grouped[(source(row), str(metadata.get("eval_protocol") or "unknown"))].append(record)
        records.append(record)
    metrics = {
        f"{source_name}/{protocol}": binary_metrics(group, "persistent")
        for (source_name, protocol), group in sorted(grouped.items())
    }
    metrics["overall"] = binary_metrics(records, "persistent")
    return {
        "model": "persistence logistic survival proxy",
        "valid_nll": valid_nll,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    train = read(args.train_jsonl)
    valid = read(args.valid_jsonl)
    test = read(args.test_jsonl)
    result = {
        "data": {
            task: {"train": len(train[task]), "valid": len(valid[task]), "test": len(test[task])}
            for task in sorted(TASKS)
        },
        "T2": train_t2(train["predict_response"], valid["predict_response"], test["predict_response"]),
        "T5": train_t5(
            train["predict_prospective_error"],
            valid["predict_prospective_error"],
            test["predict_prospective_error"],
        ),
        "T6": train_t6(
            train["predict_error_persistence"],
            valid["predict_error_persistence"],
            test["predict_error_persistence"],
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
