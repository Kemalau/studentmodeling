#!/usr/bin/env python3
"""Run lightweight source-, item-, and task-specific T1-T6 baselines."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


TASKS = [
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "estimate_mastery",
    "predict_prospective_error",
    "predict_error_persistence",
]
TASK_SHORT = {
    "predict_correctness": "T1",
    "predict_response": "T2",
    "diagnose_misconception": "T3",
    "estimate_mastery": "T4",
    "predict_prospective_error": "T5",
    "predict_error_persistence": "T6",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")


def question_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("question_id") or (row.get("query") or {}).get("question_id") or "")


def mode(counts: Counter[str], fallback: str = "") -> str:
    if not counts:
        return fallback
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def task_text(row: dict[str, Any]) -> str:
    query = row.get("query") or {}
    history = row.get("history") or []
    if source(row) == "eedi_mining":
        return str(query.get("query_text") or "")
    if source(row) == "malrule":
        parts = []
        for event in history:
            parts.append(
                f"Problem: {event.get('problem_text', '')} Student answer: {event.get('student_answer', '')} "
                f"Correct answer: {event.get('correct_answer', '')}"
            )
        parts.append(f"Problem: {query.get('problem_text', '')}")
        return "\n".join(parts)
    return str(query.get("question_text") or "")


def history_correctness(row: dict[str, Any], same_kc: bool = False) -> list[int]:
    history = row.get("history") or []
    if not same_kc:
        return [int(event.get("correctness", event.get("is_correct", 0))) for event in history]
    query_kcs = set(str(value) for value in ((row.get("query") or {}).get("kc_ids") or []))
    outcomes = []
    for event in history:
        event_kcs = set(str(value) for value in (event.get("kc_ids") or []))
        if query_kcs.intersection(event_kcs):
            outcomes.append(int(event.get("correctness", event.get("is_correct", 0))))
    return outcomes


def mastery_from_history(row: dict[str, Any]) -> str:
    outcomes = history_correctness(row, same_kc=True) or history_correctness(row)
    if not outcomes:
        return "medium"
    rate = sum(outcomes) / len(outcomes)
    if rate < 0.4:
        return "low"
    if rate < 0.7:
        return "medium"
    return "high"


def collect_train(path: Path) -> dict[str, Any]:
    global_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    question_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    t3_texts = []
    t3_labels = []
    t3_sources = []
    t3_candidate_texts = set()
    seen_t3 = set()
    for row in read_jsonl(path):
        task = str(row["task"])
        src = source(row)
        label = str(row["target_label"])
        qid = question_id(row)
        global_counts[(src, task)][label] += 1
        question_counts[(src, task, qid)][label] += 1
        if task == "diagnose_misconception":
            original_id = str((row.get("metadata") or {}).get("original_example_id") or row["example_id"])
            if original_id not in seen_t3:
                seen_t3.add(original_id)
                t3_texts.append(task_text(row))
                t3_labels.append(label)
                t3_sources.append(src)
            t3_candidate_texts.update(str(value) for value in (row.get("candidate_targets") or {}).values())
    return {
        "global_counts": global_counts,
        "question_counts": question_counts,
        "t3_texts": t3_texts,
        "t3_labels": t3_labels,
        "t3_sources": t3_sources,
        "t3_candidate_texts": t3_candidate_texts,
    }


def predict_tfidf_candidates(rows: list[dict[str, Any]], train: dict[str, Any]) -> list[str]:
    corpus = list(train["t3_texts"]) + sorted(train["t3_candidate_texts"])
    for row in rows:
        corpus.extend(str(value) for value in (row.get("candidate_targets") or {}).values())
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=120_000)
    vectorizer.fit(corpus)
    predictions = [""] * len(rows)
    malrule_train = [
        (text, label)
        for text, label, src in zip(train["t3_texts"], train["t3_labels"], train["t3_sources"])
        if src == "malrule"
    ]
    malrule_indices = [index for index, row in enumerate(rows) if source(row) == "malrule"]
    if malrule_train and malrule_indices:
        malrule_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=120_000)
        train_vectors = malrule_vectorizer.fit_transform([item[0] for item in malrule_train])
        classifier = LinearSVC(class_weight="balanced")
        classifier.fit(train_vectors, [item[1] for item in malrule_train])
        test_vectors = malrule_vectorizer.transform([task_text(rows[index]) for index in malrule_indices])
        for index, prediction in zip(malrule_indices, classifier.predict(test_vectors)):
            predictions[index] = str(prediction)

    for row_index, row in enumerate(rows):
        if source(row) == "malrule":
            continue
        labels = list(row["candidate_labels"])
        targets = [str(row["candidate_targets"][label]) for label in labels]
        query_vector = vectorizer.transform([task_text(row)])
        candidate_vectors = vectorizer.transform(targets)
        scores = (candidate_vectors @ query_vector.T).toarray().reshape(-1)
        best = max(range(len(labels)), key=lambda index: (scores[index], -index))
        predictions[row_index] = str(labels[best])
    return predictions


def specialist_prediction(row: dict[str, Any], train: dict[str, Any]) -> str:
    task = str(row["task"])
    src = source(row)
    qid = question_id(row)
    global_label = mode(train["global_counts"][(src, task)])
    item_label = mode(train["question_counts"][(src, task, qid)], global_label)
    if task == "predict_correctness":
        outcomes = history_correctness(row)
        if outcomes:
            return "correct" if sum(outcomes) / len(outcomes) >= 0.5 else "incorrect"
        return item_label
    if task in {"predict_response", "predict_prospective_error", "predict_error_persistence"}:
        return item_label
    if task == "estimate_mastery":
        return mastery_from_history(row)
    return global_label


def evaluate_predictions(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    labels = [str(row["target_label"]) for row in rows]
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def run_baselines(train_path: Path, test_path: Path) -> dict[str, Any]:
    train = collect_train(train_path)
    test_rows = list(read_jsonl(test_path))
    models = {
        "source_majority": [],
        "question_majority": [],
        "task_heuristic": [],
    }
    t3_indices = []
    t3_rows = []
    for index, row in enumerate(test_rows):
        task = str(row["task"])
        src = source(row)
        qid = question_id(row)
        fallback = mode(train["global_counts"][(src, task)])
        models["source_majority"].append(fallback)
        models["question_majority"].append(mode(train["question_counts"][(src, task, qid)], fallback))
        models["task_heuristic"].append(specialist_prediction(row, train))
        if task == "diagnose_misconception":
            t3_indices.append(index)
            t3_rows.append(row)
    for index, prediction in zip(t3_indices, predict_tfidf_candidates(t3_rows, train)):
        models["task_heuristic"][index] = prediction

    results = {"models": {}, "tasks": TASKS}
    for model_name, predictions in models.items():
        model_result = {"overall": evaluate_predictions(test_rows, predictions), "by_task": {}, "by_task_source": {}}
        for task in TASKS:
            indices = [index for index, row in enumerate(test_rows) if row["task"] == task]
            rows = [test_rows[index] for index in indices]
            preds = [predictions[index] for index in indices]
            model_result["by_task"][task] = evaluate_predictions(rows, preds)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(test_rows):
            groups[(str(row["task"]), source(row))].append(index)
        for (task, src), indices in sorted(groups.items()):
            rows = [test_rows[index] for index in indices]
            preds = [predictions[index] for index in indices]
            model_result["by_task_source"][f"{task}/{src}"] = evaluate_predictions(rows, preds)
        results["models"][model_name] = model_result
    return results


def markdown_matrix(results: dict[str, Any]) -> str:
    lines = [
        "# T1-T6 Evaluation Matrix",
        "",
        "Accuracy is shown below. Macro-F1 and source-level breakdowns are stored in `simple_baseline_results.json`.",
        "",
        "| Model | T1 | T2 | T3 | T4 | T5 | T6 | Macro Avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, model in results["models"].items():
        values = [float(model["by_task"][task]["accuracy"]) for task in TASKS]
        cells = " | ".join(f"{value:.3f}" for value in values)
        lines.append(f"| {model_name} | {cells} | {sum(values) / len(values):.3f} |")
    if not any(model_name.startswith("Qwen3-32B") for model_name in results["models"]):
        lines.append("| Qwen3-32B mixed QLoRA | pending | pending | pending | pending | pending | pending | pending |")
    lines.extend(
        [
            "",
            "## Baseline Definitions",
            "",
            "- `source_majority`: most frequent label for each source/task.",
            "- `question_majority`: most frequent label for each source/task/question, with source-majority backoff.",
            "- `task_heuristic`: history correctness for T1, question response mode for T2/T5, TF-IDF candidate retrieval for T3, same-KC history rate for T4, and question persistence prior for T6. This is a sanity baseline, not a traditional SOTA model.",
        ]
    )
    lines.extend(["", "## Source-Level Accuracy", ""])
    source_keys = sorted(next(iter(results["models"].values()))["by_task_source"])
    lines.append("| Task / Source | " + " | ".join(results["models"]) + " |")
    lines.append("|---|" + "---:|" * len(results["models"]))
    for key in source_keys:
        cells = []
        for model in results["models"]:
            metrics = results["models"][model]["by_task_source"].get(key)
            cells.append("-" if metrics is None else f"{metrics['accuracy']:.3f}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def svg_matrix(results: dict[str, Any]) -> str:
    model_names = list(results["models"])
    if not any(model_name.startswith("Qwen3-32B") for model_name in model_names):
        model_names.append("Qwen3-32B QLoRA")
    width, left, top, cell_w, cell_h = 960, 250, 70, 105, 52
    height = top + len(model_names) * cell_h + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.head{font-size:16px;font-weight:700}.label{font-size:14px}.value{font-size:15px;font-weight:700}</style>',
        '<text x="24" y="32" class="head">T1-T6 Evaluation Matrix (Accuracy)</text>',
    ]
    for column, task in enumerate(TASKS):
        parts.append(f'<text x="{left + column * cell_w + cell_w / 2}" y="58" text-anchor="middle" class="head">{TASK_SHORT[task]}</text>')
    for row_index, model_name in enumerate(model_names):
        y = top + row_index * cell_h
        parts.append(f'<text x="24" y="{y + 32}" class="label">{html.escape(model_name)}</text>')
        for column, task in enumerate(TASKS):
            x = left + column * cell_w
            if model_name == "Qwen3-32B QLoRA":
                fill, value = "#e5e7eb", "pending"
            else:
                score = float(results["models"][model_name]["by_task"][task]["accuracy"])
                red = int(245 - 100 * score)
                green = int(247 - 15 * score)
                blue = int(250 - 120 * score)
                fill, value = f"rgb({red},{green},{blue})", f"{score:.3f}"
            parts.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="7" fill="{fill}"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 32}" text-anchor="middle" class="value">{value}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    results = run_baselines(args.train_jsonl, args.test_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "simple_baseline_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "evaluation_matrix.md").write_text(markdown_matrix(results), encoding="utf-8")
    (args.out_dir / "evaluation_matrix.svg").write_text(svg_matrix(results), encoding="utf-8")
    print(markdown_matrix(results))


if __name__ == "__main__":
    main()
