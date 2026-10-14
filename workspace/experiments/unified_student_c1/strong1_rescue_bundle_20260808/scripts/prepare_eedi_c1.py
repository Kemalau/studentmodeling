#!/usr/bin/env python3
"""Prepare C1 unified-generation samples from canonical Eedi interactions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LETTERS = ["A", "B", "C", "D"]
TASKS = ["next_correctness", "next_response", "future_mastery"]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_canonical(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("student_id") or not row.get("question_id"):
                continue
            try:
                row["correctness"] = int(float(row.get("correctness", "0")))
            except ValueError:
                continue
            if row.get("student_response") not in LETTERS:
                continue
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def parse_options(row: dict[str, Any]) -> dict[str, str]:
    raw = row.get("options") or ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {letter: f"Option {letter}" for letter in LETTERS}
    if not isinstance(obj, dict):
        return {letter: f"Option {letter}" for letter in LETTERS}
    return {str(k): str(v) for k, v in obj.items() if str(k) in LETTERS}


def normalize_interaction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "student_id": str(row["student_id"]),
        "timestamp": str(row.get("timestamp") or ""),
        "question_id": str(row["question_id"]),
        "skill_id": str(row.get("skill_id") or "unknown"),
        "skill_name": str(row.get("skill_name") or "unknown"),
        "question_text": str(row.get("question_text") or f"Question {row['question_id']}"),
        "options": parse_options(row),
        "student_response": str(row.get("student_response") or ""),
        "correct_response": str(row.get("correct_response") or ""),
        "correctness": int(row["correctness"]),
    }


def split_students(student_ids: list[str], seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    ids = list(student_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    return {
        "train": set(ids[:n_train]),
        "valid": set(ids[n_train : n_train + n_valid]),
        "test": set(ids[n_train + n_valid :]),
    }


def format_history(history: list[dict[str, Any]]) -> str:
    parts = []
    for i, trial in enumerate(history, 1):
        parts.append(
            "\n".join(
                [
                    f"Trial {i}:",
                    f"Skill: {trial['skill_name']}",
                    f"Question: {trial['question_text']}",
                    f"Student answer: {trial['student_response']}",
                    f"Correct answer: {trial['correct_response']}",
                    f"Outcome: {'correct' if trial['correctness'] else 'incorrect'}",
                ]
            )
        )
    return "\n\n".join(parts)


def format_query(query: dict[str, Any], include_options: bool = True) -> str:
    lines = [
        f"Skill: {query['skill_name']}",
        f"Question: {query['question_text']}",
        f"Known correct answer: {query['correct_response']}",
    ]
    if include_options:
        lines.append("Allowed answers: A, B, C, D")
    return "\n".join(lines)


def output_tag(task: str, label: str) -> str:
    if task == "next_correctness":
        return f"<correctness>{label}</correctness>"
    if task == "next_response":
        return f"<predicted_answer>{label}</predicted_answer>"
    if task == "future_mastery":
        return f"<mastery>{label}</mastery>"
    raise ValueError(task)


def labels_for_task(task: str) -> list[str]:
    if task == "next_correctness":
        return ["correct", "incorrect"]
    if task == "next_response":
        return LETTERS
    if task == "future_mastery":
        return ["low", "medium", "high"]
    raise ValueError(task)


def instruction_for_task(task: str) -> str:
    if task == "next_correctness":
        return "Predict whether this student will answer the next question correctly."
    if task == "next_response":
        return "Predict the option letter this student will choose on the next question."
    if task == "future_mastery":
        return (
            "Estimate the student's near-future mastery on the queried skill, "
            "using low, medium, or high."
        )
    raise ValueError(task)


def build_prompt(task: str, history: list[dict[str, Any]], query: dict[str, Any]) -> str:
    candidates = ", ".join(output_tag(task, label) for label in labels_for_task(task))
    return (
        "You are a student modeling system for middle-school mathematics.\n"
        f"<task>{task}</task>\n"
        f"<instruction>{instruction_for_task(task)} Return exactly one valid XML-style tag.</instruction>\n"
        "<student_history>\n"
        f"{format_history(history)}\n"
        "</student_history>\n"
        "<STUDENT_STATE>\n"
        "<query>\n"
        f"{format_query(query)}\n"
        "</query>\n"
        f"Valid outputs: {candidates}\n"
        "Answer:"
    )


def future_mastery_label(traces: list[dict[str, Any]], idx: int, future_n: int) -> str | None:
    skill = traces[idx]["skill_id"]
    future = []
    for trial in traces[idx:]:
        if trial["skill_id"] == skill:
            future.append(int(trial["correctness"]))
        if len(future) >= future_n:
            break
    if len(future) < max(2, future_n // 2):
        return None
    rate = sum(future) / len(future)
    if rate < 0.4:
        return "low"
    if rate < 0.7:
        return "medium"
    return "high"


def make_example(
    dataset: str,
    split: str,
    task: str,
    student_id: str,
    traces: list[dict[str, Any]],
    idx: int,
    history_k: int,
    future_n: int,
) -> dict[str, Any] | None:
    history = traces[max(0, idx - history_k) : idx]
    query = dict(traces[idx])
    query.pop("student_response", None)
    if task == "next_correctness":
        label = "correct" if traces[idx]["correctness"] else "incorrect"
    elif task == "next_response":
        label = traces[idx]["student_response"]
    elif task == "future_mastery":
        label = future_mastery_label(traces, idx, future_n)
        if label is None:
            return None
    else:
        raise ValueError(task)

    candidates = labels_for_task(task)
    target = output_tag(task, label)
    return {
        "example_id": f"{dataset}:{task}:{split}:{student_id}:{idx}:{traces[idx]['question_id']}",
        "dataset": dataset,
        "split": split,
        "task": task,
        "student_id": student_id,
        "history": history,
        "query": query,
        "target_label": label,
        "target": target,
        "candidate_labels": candidates,
        "candidate_targets": {cand: output_tag(task, cand) for cand in candidates},
        "prompt": build_prompt(task, history, query),
        "metadata": {
            "question_id": traces[idx]["question_id"],
            "skill_id": traces[idx]["skill_id"],
            "skill_name": traces[idx]["skill_name"],
            "target_correctness": traces[idx]["correctness"],
            "target_response": traces[idx]["student_response"],
        },
    }


def sample_indices(
    traces: list[dict[str, Any]],
    max_targets: int,
    rng: random.Random,
    balance: bool,
) -> list[int]:
    candidates = list(range(1, len(traces)))
    if len(candidates) <= max_targets:
        return candidates
    if not balance:
        rng.shuffle(candidates)
        return sorted(candidates[:max_targets])
    correct = [i for i in candidates if traces[i]["correctness"] == 1]
    incorrect = [i for i in candidates if traces[i]["correctness"] == 0]
    rng.shuffle(correct)
    rng.shuffle(incorrect)
    half = max_targets // 2
    chosen = incorrect[:half] + correct[: max_targets - min(half, len(incorrect))]
    if len(chosen) < max_targets:
        used = set(chosen)
        rest = [i for i in candidates if i not in used]
        rng.shuffle(rest)
        chosen.extend(rest[: max_targets - len(chosen)])
    return sorted(chosen[:max_targets])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split_task = Counter((row["split"], row["task"]) for row in rows)
    label_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for task in TASKS:
        counts = Counter(row["target_label"] for row in rows if row["task"] == task)
        label_counts[task] = dict(counts)
    return {
        "n_examples": len(rows),
        "by_split_task": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_split_task.items())},
        "label_counts": label_counts,
        "students": len({row["student_id"] for row in rows}),
        "questions": len({row["metadata"]["question_id"] for row in rows}),
        "skills": len({row["metadata"]["skill_id"] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="eedi_neurips")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-k", type=int, default=12)
    parser.add_argument("--future-n", type=int, default=5)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-train-per-student", type=int, default=8)
    parser.add_argument("--max-eval-per-student", type=int, default=4)
    args = parser.parse_args()

    rows = [normalize_interaction(row) for row in read_canonical(args.canonical_csv, args.max_rows)]
    by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_student[row["student_id"]].append(row)
    for traces in by_student.values():
        traces.sort(key=lambda r: (r["timestamp"], r["question_id"]))
    by_student = {sid: traces for sid, traces in by_student.items() if len(traces) >= 3}

    splits = split_students(sorted(by_student), args.seed)
    rng = random.Random(args.seed)
    out_rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for split, students in splits.items():
        max_targets = args.max_train_per_student if split == "train" else args.max_eval_per_student
        for student_id in sorted(students):
            traces = by_student[student_id]
            indices = sample_indices(traces, max_targets, rng, balance=True)
            for idx in indices:
                for task in TASKS:
                    example = make_example(
                        args.dataset_name,
                        split,
                        task,
                        student_id,
                        traces,
                        idx,
                        args.history_k,
                        args.future_n,
                    )
                    if example is not None:
                        out_rows[split].append(example)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for split, split_rows in out_rows.items():
        split_rows.sort(key=lambda r: r["example_id"])
        all_rows.extend(split_rows)
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)
        for task in TASKS:
            write_jsonl(args.out_dir / f"{split}_{task}.jsonl", [r for r in split_rows if r["task"] == task])
    stats = summarize(all_rows)
    stats["source_csv"] = str(args.canonical_csv)
    stats["history_k"] = args.history_k
    stats["future_n"] = args.future_n
    stats["students_by_split"] = {split: len(students) for split, students in splits.items()}
    (args.out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
