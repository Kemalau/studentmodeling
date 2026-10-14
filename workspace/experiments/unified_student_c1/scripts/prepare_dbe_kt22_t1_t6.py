#!/usr/bin/env python3
"""Prepare DBE-KT22 for T1, T2, T4, T5, and strict same-error T6."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from student_task_utils import (
    choose_positions,
    clean_text,
    make_row,
    mastery_label,
    meaningful_text_length,
    output_tag,
    split_students,
    summarize,
    write_dataset,
)


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_traces(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    questions = {row["id"]: row for row in read_table(raw_dir / "Questions.csv")}
    choices = {row["id"]: row for row in read_table(raw_dir / "Question_Choices.csv")}
    choices_by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
    for choice in choices.values():
        choices_by_question[choice["question_id"]].append(choice)
    kcs = {row["id"]: row for row in read_table(raw_dir / "KCs.csv")}
    question_kcs: dict[str, list[str]] = defaultdict(list)
    for row in read_table(raw_dir / "Question_KC_Relationships.csv"):
        question_kcs[row["question_id"]].append(row["knowledgecomponent_id"])

    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_table(raw_dir / "Transaction.csv"):
        if row["is_hidden"].lower() == "true" or row["answer_state"].lower() not in {"true", "false"}:
            continue
        question = questions.get(row["question_id"])
        selected = choices.get(row["answer_choice_id"])
        if question is None or selected is None:
            continue
        available = sorted(choices_by_question[row["question_id"]], key=lambda item: int(item["id"]))
        correct = next((item for item in available if item["is_correct"].lower() == "true"), None)
        kc_ids = question_kcs[row["question_id"]]
        question_text = clean_text(question["question_text"] or question["question_rich_text"])
        if meaningful_text_length(question_text) < 8:
            continue
        event = {
            "interaction_id": row["id"],
            "timestamp": row["start_time"],
            "question_id": row["question_id"],
            "question_text": question_text,
            "kc_ids": kc_ids,
            "kc_names": [clean_text(kcs[kc_id]["name"]) for kc_id in kc_ids],
            "options": {item["id"]: clean_text(item["choice_text"]) for item in available},
            "student_response_id": selected["id"],
            "student_response": clean_text(selected["choice_text"]),
            "correct_response_id": correct["id"] if correct else "",
            "correct_response": clean_text(correct["choice_text"]) if correct else "",
            "correctness": int(row["answer_state"].lower() == "true"),
            "hint_used": row["hint_used"].lower() == "true",
        }
        traces[row["student_id"]].append(event)
    for trace in traces.values():
        trace.sort(key=lambda event: (event["timestamp"], int(event["interaction_id"])))
    return {student_id: trace for student_id, trace in traces.items() if len(trace) >= 3}


def response_candidates(event: dict[str, Any]) -> dict[str, str]:
    return {choice_id: output_tag("predict_response", text) for choice_id, text in event["options"].items()}


def prospective_candidates(event: dict[str, Any]) -> dict[str, str]:
    candidates = {}
    for choice_id, text in event["options"].items():
        correctness = "correct" if choice_id == event["correct_response_id"] else "incorrect"
        candidates[choice_id] = (
            f"<correctness>{correctness}</correctness>"
            f"<predicted_response>{text}</predicted_response>"
        )
    return candidates


def metadata(event: dict[str, Any], **extra: Any) -> dict[str, Any]:
    result = {
        "source": "DBE-KT22",
        "question_id": event["question_id"],
        "skill_id": "|".join(event["kc_ids"]),
        "skill_name": "|".join(event["kc_names"]),
        "timestamp": event["timestamp"],
        "target_correctness": event["correctness"],
        "target_response_id": event["student_response_id"],
        "target_response": event["student_response"],
    }
    result.update(extra)
    return result


def make_rows_for_student(
    split: str,
    student_id: str,
    trace: list[dict[str, Any]],
    positions: list[int],
    history_k: int,
    future_n: int,
    prospective_delta: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    binary = {label: output_tag("predict_correctness", label) for label in ("correct", "incorrect")}
    mastery = {label: output_tag("estimate_mastery", label) for label in ("low", "medium", "high")}
    persistence = {label: output_tag("predict_error_persistence", label) for label in ("persistent", "transient")}
    for idx in positions:
        event = trace[idx]
        history = trace[max(0, idx - history_k):idx]
        correctness = "correct" if event["correctness"] else "incorrect"
        common = dict(dataset="dbe_kt22", split=split, student_id=student_id, position=idx, history=history, query=event)
        rows.append(make_row(task="predict_correctness", target_label=correctness, candidate_targets=binary, metadata=metadata(event), **common))
        rows.append(make_row(task="predict_response", target_label=event["student_response_id"], candidate_targets=response_candidates(event), metadata=metadata(event), **common))

        mastery_result = mastery_label(trace, idx, future_n)
        if mastery_result:
            label, outcomes = mastery_result
            rows.append(make_row(task="estimate_mastery", target_label=label, candidate_targets=mastery, metadata=metadata(event, future_same_kc_outcomes=outcomes), **common))

        future_idx = idx + prospective_delta
        if future_idx < len(trace):
            future_event = trace[future_idx]
            prospective_history = trace[max(0, idx - history_k + 1):idx + 1]
            rows.append(
                make_row(
                    dataset="dbe_kt22",
                    split=split,
                    task="predict_prospective_error",
                    student_id=student_id,
                    position=future_idx,
                    history=prospective_history,
                    query=future_event,
                    target_label=future_event["student_response_id"],
                    candidate_targets=prospective_candidates(future_event),
                    metadata=metadata(future_event, prospective_delta=prospective_delta, history_cutoff_position=idx),
                    prospective_delta=prospective_delta,
                )
            )

    for idx, event in enumerate(trace):
        if event["correctness"]:
            continue
        future_same_question = [
            candidate for candidate in trace[idx + 1:] if candidate["question_id"] == event["question_id"]
        ][:3]
        if not future_same_question:
            continue
        repeats = any(candidate["student_response_id"] == event["student_response_id"] for candidate in future_same_question)
        label = "persistent" if repeats else "transient"
        history = trace[max(0, idx - history_k):idx]
        rows.append(
            make_row(
                dataset="dbe_kt22",
                split=split,
                task="predict_error_persistence",
                student_id=student_id,
                position=idx,
                history=history,
                query=event,
                target_label=label,
                candidate_targets=persistence,
                metadata=metadata(
                    event,
                    persistence_definition="same question and same wrong choice recurs within next 3 same-question attempts",
                    future_same_question_attempts=len(future_same_question),
                ),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/unified_student_c1/data/raw/dbe_kt22"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/dbe_kt22_t1_t6"))
    parser.add_argument("--history-k", type=int, default=12)
    parser.add_argument("--future-n", type=int, default=5)
    parser.add_argument("--prospective-delta", type=int, default=3)
    parser.add_argument("--max-train-per-student", type=int, default=8)
    parser.add_argument("--max-eval-per-student", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    traces = load_traces(args.raw_dir)
    splits = split_students(sorted(traces), args.seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    rng = random.Random(args.seed)
    for split, students in splits.items():
        maximum = args.max_train_per_student if split == "train" else args.max_eval_per_student
        for student_id in sorted(students):
            trace = traces[student_id]
            positions = choose_positions(trace, maximum, rng)
            rows_by_split[split].extend(
                make_rows_for_student(split, student_id, trace, positions, args.history_k, args.future_n, args.prospective_delta)
            )
    stats = summarize(rows_by_split)
    stats.update({
        "source_dir": str(args.raw_dir),
        "history_k": args.history_k,
        "future_n": args.future_n,
        "prospective_delta": args.prospective_delta,
        "students_by_split": {split: len(students) for split, students in splits.items()},
        "task_scope": {
            "predict_error_persistence": "strict same-question, same-wrong-choice recurrence",
        },
    })
    write_dataset(args.out_dir, rows_by_split, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
