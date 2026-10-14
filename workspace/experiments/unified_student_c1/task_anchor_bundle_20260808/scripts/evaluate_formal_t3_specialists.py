#!/usr/bin/env python3
"""Evaluate full-taxonomy T3 specialist baselines on the formal manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from formal_metric_utils import extract_tag


def read(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("task") == "diagnose_misconception":
                    rows.append(row)
    return rows


def source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("eval_source") or metadata.get("mix_source") or row.get("dataset") or "unknown")


def target_text(row: dict[str, Any]) -> str:
    parsed = extract_tag(str(row.get("target") or ""), "misconception")
    return str(parsed if parsed is not None else row.get("target_label") or "")


def taxonomy_texts(row: dict[str, Any]) -> set[str]:
    values = {target_text(row)}
    for target in (row.get("candidate_targets") or {}).values():
        parsed = extract_tag(str(target), "misconception")
        if parsed:
            values.add(parsed)
    return {value for value in values if value}


def row_text(row: dict[str, Any]) -> str:
    if source(row) == "eedi_mining":
        return str((row.get("query") or {}).get("query_text") or "")
    parts = ["source malrule"]
    for event in row.get("history") or []:
        parts.extend(
            [
                str(event.get("problem_text") or ""),
                f"student answer {event.get('student_answer', '')}",
                f"correct answer {event.get('correct_answer', '')}",
                f"operation {event.get('operation', '')}",
            ]
        )
    query = row.get("query") or {}
    parts.extend([str(query.get("problem_text") or ""), str(query.get("operation") or "")])
    return "\n".join(parts)


def classification_metrics(labels: list[str], predictions: list[str], known: set[str]) -> dict[str, Any]:
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "taxonomy_coverage": sum(label in known for label in labels) / max(1, len(labels)),
        "n_taxonomy_labels": len(known),
    }


def train_tfidf(
    train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_source_train: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_valid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_source_train[source(row)].append(row)
    for row in valid_rows:
        by_source_valid[source(row)].append(row)
    for row in test_rows:
        by_source_test[source(row)].append(row)

    result = {"model": "TF-IDF bigram + LinearSVC full-taxonomy classifier", "by_source": {}}
    all_labels, all_predictions = [], []
    all_known: set[str] = set()
    for source_name, source_test in sorted(by_source_test.items()):
        source_train = by_source_train[source_name]
        source_valid = by_source_valid[source_name]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=200000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        x_train = vectorizer.fit_transform([row_text(row) for row in source_train])
        y_train = [target_text(row) for row in source_train]
        x_valid = vectorizer.transform([row_text(row) for row in source_valid])
        y_valid = [target_text(row) for row in source_valid]
        best = None
        for regularization in (0.25, 0.5, 1.0, 2.0):
            model = LinearSVC(C=regularization, random_state=42)
            model.fit(x_train, y_train)
            predictions = model.predict(x_valid)
            score = f1_score(y_valid, predictions, average="macro", zero_division=0)
            if best is None or score > best[0]:
                best = (score, regularization, model)
        assert best is not None
        source_predictions = best[2].predict(vectorizer.transform([row_text(row) for row in source_test])).tolist()
        source_labels = [target_text(row) for row in source_test]
        known = set(y_train) | {target_text(row) for row in source_valid}
        metrics = classification_metrics(source_labels, source_predictions, known)
        metrics.update({"selected_c": best[1], "valid_macro_f1": best[0]})
        result["by_source"][source_name] = metrics
        all_labels.extend(source_labels)
        all_predictions.extend(source_predictions)
        all_known.update(known)
    result["overall"] = classification_metrics(all_labels, all_predictions, all_known)
    return result


def bge_full_taxonomy(
    train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    taxonomy = sorted(
        {
            label
            for row in train_rows + valid_rows
            if source(row) == "eedi_mining"
            for label in taxonomy_texts(row)
        }
    )
    rows = [row for row in test_rows if source(row) == "eedi_mining"]
    model = SentenceTransformer(args.bge_model, device=args.bge_device)
    candidate_embeddings = model.encode(
        taxonomy,
        batch_size=args.bge_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        ["Represent this sentence for searching relevant passages: " + row_text(row) for row in rows],
        batch_size=args.bge_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    scores = query_embeddings @ candidate_embeddings.T
    order = np.argsort(-scores, axis=1)
    predictions = [taxonomy[int(indices[0])] for indices in order]
    labels = [target_text(row) for row in rows]
    taxonomy_index = {label: index for index, label in enumerate(taxonomy)}
    reciprocal_ranks = []
    recall = {1: 0, 5: 0, 10: 0}
    for label, indices in zip(labels, order):
        if label not in taxonomy_index:
            reciprocal_ranks.append(0.0)
            continue
        rank = int(np.where(indices == taxonomy_index[label])[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        for cutoff in recall:
            recall[cutoff] += int(rank <= cutoff)
    result = classification_metrics(labels, predictions, set(taxonomy))
    result.update(
        {
            "model": args.bge_model,
            "protocol": "full train-taxonomy retrieval; no oracle candidates",
            "mrr": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
            **{f"recall_at_{cutoff}": value / max(1, len(rows)) for cutoff, value in recall.items()},
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bge-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--bge-device", default="cpu")
    parser.add_argument("--bge-batch-size", type=int, default=64)
    args = parser.parse_args()

    train_rows = read(args.train_jsonl)
    valid_rows = read(args.valid_jsonl)
    test_rows = read(args.test_jsonl)
    result = {
        "data": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "tfidf_linearsvc": train_tfidf(train_rows, valid_rows, test_rows),
        "bge_full_taxonomy_eedi": bge_full_taxonomy(train_rows, valid_rows, test_rows, args),
        "metric_note": "Exact taxonomy-label match; Qwen semantic remapping remains pending.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
