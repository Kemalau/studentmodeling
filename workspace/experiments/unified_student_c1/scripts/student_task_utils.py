#!/usr/bin/env python3
"""Shared helpers for T1-T6 student-modeling dataset preparation."""

from __future__ import annotations

import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATE_TOKEN = "<STUDENT_STATE>"
CANONICAL_TASKS = [
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "estimate_mastery",
    "predict_prospective_error",
    "predict_error_persistence",
]


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<img[^>]*src=[\"']([^\"']+)[\"'][^>]*>", r" [formula/image: \1] ", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:p|div|center|table|tr|td|th|ul|ol|li|span|strong|em|b|i)[^>]*>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return " ".join(text.split())


def meaningful_text_length(value: Any) -> int:
    text = re.sub(r"(?:question|analysis)_\d+-image_\d+", " ", str(value or ""), flags=re.I)
    text = re.sub(r"https?://\S+", " ", text)
    return len(re.findall(r"[A-Za-z0-9\u3400-\u9fff]", text))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_students(student_ids: list[str], seed: int) -> dict[str, set[str]]:
    ids = list(student_ids)
    random.Random(seed).shuffle(ids)
    train_end = int(0.8 * len(ids))
    valid_end = int(0.9 * len(ids))
    return {
        "train": set(ids[:train_end]),
        "valid": set(ids[train_end:valid_end]),
        "test": set(ids[valid_end:]),
    }


def mastery_label(trace: list[dict[str, Any]], idx: int, future_n: int) -> tuple[str, list[int]] | None:
    query_kcs = set(trace[idx]["kc_ids"])
    outcomes: list[int] = []
    for event in trace[idx + 1:]:
        if query_kcs.intersection(event["kc_ids"]):
            outcomes.append(int(event["correctness"]))
        if len(outcomes) >= future_n:
            break
    if len(outcomes) < max(2, future_n // 2):
        return None
    rate = sum(outcomes) / len(outcomes)
    if rate < 0.4:
        return "low", outcomes
    if rate < 0.7:
        return "medium", outcomes
    return "high", outcomes


def output_tag(task: str, label: str) -> str:
    if task == "predict_correctness":
        return f"<correctness>{label}</correctness>"
    if task == "predict_response":
        return f"<predicted_response>{label}</predicted_response>"
    if task == "estimate_mastery":
        return f"<mastery>{label}</mastery>"
    if task == "predict_error_persistence":
        return f"<persistence>{label}</persistence>"
    raise ValueError(task)


def format_event(event: dict[str, Any], number: int) -> str:
    lines = [
        f"Trial {number}:",
        f"Knowledge components: {', '.join(event['kc_names'])}",
        f"Question: {event['question_text']}",
    ]
    if event.get("student_response"):
        lines.append(f"Student response: {event['student_response']}")
    if event.get("correct_response"):
        lines.append(f"Correct response: {event['correct_response']}")
    lines.append(f"Outcome: {'correct' if event['correctness'] else 'incorrect'}")
    return "\n".join(lines)


def format_query(query: dict[str, Any], task: str) -> str:
    lines = [
        f"Knowledge components: {', '.join(query['kc_names'])}",
        f"Question: {query['question_text']}",
    ]
    if query.get("options"):
        lines.append("Answer choices:")
        lines.extend(f"- {key}: {value}" for key, value in query["options"].items())
    if task == "predict_error_persistence":
        lines.append(f"Observed error response: {query.get('student_response') or 'incorrect response'}")
    return "\n".join(lines)


def instruction(task: str, prospective_delta: int | None = None) -> str:
    if task == "predict_correctness":
        return "Predict whether the student will answer the queried question correctly."
    if task == "predict_response":
        return "Predict the exact answer choice the student will select on the queried question."
    if task == "estimate_mastery":
        return "Estimate the student's near-future mastery of the queried knowledge component."
    if task == "predict_prospective_error":
        return (
            f"The queried question occurs {prospective_delta} interactions after the visible history. "
            "Predict both correctness and the exact answer choice; the intervening interactions are hidden."
        )
    if task == "predict_error_persistence":
        return "Predict whether the observed error pattern will recur on a future attempt of the same question."
    raise ValueError(task)


def build_prompt(
    task: str,
    history: list[dict[str, Any]],
    query: dict[str, Any],
    candidate_targets: dict[str, str],
    prospective_delta: int | None = None,
) -> str:
    history_text = "\n\n".join(format_event(event, idx) for idx, event in enumerate(history, 1))
    valid = "\n".join(candidate_targets.values())
    return (
        "You are a student modeling system.\n"
        f"<task>{task}</task>\n"
        f"<instruction>{instruction(task, prospective_delta)} Return exactly one valid output.</instruction>\n"
        "<student_history>\n"
        f"{history_text}\n"
        "</student_history>\n"
        f"{STATE_TOKEN}\n"
        "<query>\n"
        f"{format_query(query, task)}\n"
        "</query>\n"
        f"Valid outputs:\n{valid}\n"
        "Answer:"
    )


def make_row(
    *,
    dataset: str,
    split: str,
    task: str,
    student_id: str,
    position: int,
    history: list[dict[str, Any]],
    query: dict[str, Any],
    target_label: str,
    candidate_targets: dict[str, str],
    metadata: dict[str, Any],
    prospective_delta: int | None = None,
) -> dict[str, Any]:
    return {
        "example_id": f"{dataset}:{split}:{task}:{student_id}:{position}",
        "dataset": dataset,
        "split": split,
        "task": task,
        "student_id": student_id,
        "history": history,
        "query": query,
        "target_label": target_label,
        "target": candidate_targets[target_label],
        "candidate_labels": list(candidate_targets),
        "candidate_targets": candidate_targets,
        "prompt": build_prompt(task, history, query, candidate_targets, prospective_delta),
        "metadata": metadata,
    }


def choose_positions(trace: list[dict[str, Any]], maximum: int, rng: random.Random) -> list[int]:
    positions = list(range(1, len(trace)))
    if len(positions) > maximum:
        positive = [idx for idx in positions if trace[idx]["correctness"]]
        negative = [idx for idx in positions if not trace[idx]["correctness"]]
        rng.shuffle(positive)
        rng.shuffle(negative)
        half = (maximum + 1) // 2
        selected = negative[:half] + positive[: maximum - min(half, len(negative))]
        if len(selected) < maximum:
            used = set(selected)
            remaining = [idx for idx in positions if idx not in used]
            rng.shuffle(remaining)
            selected.extend(remaining[: maximum - len(selected)])
        positions = selected
    return sorted(positions)


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    for task in CANONICAL_TASKS:
        labels[task] = dict(Counter(row["target_label"] for row in all_rows if row["task"] == task))
    return {
        "rows_by_split": {split: len(rows) for split, rows in rows_by_split.items()},
        "rows_by_task": dict(Counter(row["task"] for row in all_rows)),
        "labels_by_task": labels,
        "students": len({row["student_id"] for row in all_rows}),
        "questions": len({str(row["metadata"].get("question_id", "")) for row in all_rows}),
    }


def write_dataset(out_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]], stats: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: row["example_id"])
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        for task in CANONICAL_TASKS:
            task_rows = [row for row in rows if row["task"] == task]
            if task_rows:
                write_jsonl(out_dir / f"{split}_{task}.jsonl", task_rows)
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
