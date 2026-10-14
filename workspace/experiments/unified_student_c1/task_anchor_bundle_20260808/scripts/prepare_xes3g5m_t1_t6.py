#!/usr/bin/env python3
"""Prepare XES3G5M question sequences for T1, T4, and response-limited T6."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
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
    summarize,
    write_dataset,
)


def split_values(value: str) -> list[str]:
    return value.split(",") if value else []


def load_questions(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads((metadata_dir / "questions.json").read_text(encoding="utf-8"))
    questions = {}
    for question_id, row in raw.items():
        raw_content = str(row.get("content") or "")
        raw_options = json.dumps(row.get("options") or {}, ensure_ascii=False)
        if re.search(
            r"(?:question|analysis)_\d+-image_\d+",
            f"{raw_content}\n{raw_options}",
            flags=re.I,
        ):
            continue
        question_text = clean_text(raw_content or f"Question {question_id}")
        if meaningful_text_length(question_text) < 4:
            continue
        routes = [clean_text(route) for route in row.get("kc_routes", [])]
        options = row.get("options") or {}
        questions[str(question_id)] = {
            "question_text": question_text,
            "kc_names": routes,
            "options": {str(key): clean_text(value) for key, value in options.items()},
            "correct_response": " | ".join(clean_text(answer) for answer in row.get("answer", [])),
            "question_type": clean_text(row.get("type")),
        }
    return questions


def read_sequences(path: Path, questions: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            fields = [split_values(row[key]) for key in ("questions", "concepts", "responses", "timestamps")]
            masks = split_values(row.get("selectmasks", ""))
            if masks:
                fields.append(masks)
            limit = min(map(len, fields))
            for idx in range(limit):
                question_id, concepts, response, timestamp = (field[idx] for field in fields[:4])
                selected = fields[4][idx] if len(fields) == 5 else "1"
                if question_id == "-1" or response not in {"0", "1"} or selected != "1":
                    continue
                question = questions.get(question_id)
                if question is None:
                    continue
                kc_ids = concepts.split("_")
                event = {
                    "timestamp": int(timestamp),
                    "question_id": question_id,
                    "question_text": question["question_text"],
                    "kc_ids": kc_ids,
                    "kc_names": question["kc_names"] or kc_ids,
                    "options": question["options"],
                    "student_response": "",
                    "correct_response": question["correct_response"],
                    "correctness": int(response),
                    "question_type": question["question_type"],
                }
                traces[row["uid"]].append(event)
    for trace in traces.values():
        trace.sort(key=lambda event: (event["timestamp"], event["question_id"]))
    return {student_id: trace for student_id, trace in traces.items() if len(trace) >= 3}


def make_rows(
    dataset_split: str,
    student_id: str,
    trace: list[dict[str, Any]],
    positions: list[int],
    history_k: int,
    future_n: int,
) -> list[dict[str, Any]]:
    rows = []
    binary = {label: output_tag("predict_correctness", label) for label in ("correct", "incorrect")}
    mastery = {label: output_tag("estimate_mastery", label) for label in ("low", "medium", "high")}
    persistence = {label: output_tag("predict_error_persistence", label) for label in ("persistent", "transient")}
    for idx in positions:
        event = trace[idx]
        history = trace[max(0, idx - history_k):idx]
        base_metadata = {
            "source": "XES3G5M",
            "question_id": event["question_id"],
            "skill_id": "|".join(event["kc_ids"]),
            "skill_name": "|".join(event["kc_names"]),
            "timestamp": event["timestamp"],
            "target_correctness": event["correctness"],
            "response_availability": "correctness only; selected answer is not recorded",
        }
        common = dict(
            dataset="xes3g5m",
            split=dataset_split,
            student_id=student_id,
            position=idx,
            history=history,
            query=event,
        )
        correctness = "correct" if event["correctness"] else "incorrect"
        rows.append(make_row(task="predict_correctness", target_label=correctness, candidate_targets=binary, metadata=base_metadata, **common))

        mastery_result = mastery_label(trace, idx, future_n)
        if mastery_result:
            label, outcomes = mastery_result
            rows.append(
                make_row(
                    task="estimate_mastery",
                    target_label=label,
                    candidate_targets=mastery,
                    metadata={**base_metadata, "future_same_kc_outcomes": outcomes},
                    **common,
                )
            )

    persistence_positions = []
    for idx, event in enumerate(trace):
        if event["correctness"]:
            continue
        future_same_question = [
            candidate for candidate in trace[idx + 1:] if candidate["question_id"] == event["question_id"]
        ][:3]
        if future_same_question:
            persistence_positions.append((idx, future_same_question))
    if len(persistence_positions) > 2:
        persistence_positions = [persistence_positions[0], persistence_positions[-1]]
    for idx, future_same_question in persistence_positions:
        event = trace[idx]
        label = "persistent" if any(not candidate["correctness"] for candidate in future_same_question) else "transient"
        history = trace[max(0, idx - history_k):idx]
        base_metadata = {
            "source": "XES3G5M",
            "question_id": event["question_id"],
            "skill_id": "|".join(event["kc_ids"]),
            "skill_name": "|".join(event["kc_names"]),
            "timestamp": event["timestamp"],
            "target_correctness": event["correctness"],
            "response_availability": "correctness only; selected answer is not recorded",
            "persistence_definition": "same question is answered incorrectly again within next 3 same-question attempts",
            "limitation": "XES3G5M does not record the selected answer, so exact wrong-response recurrence is unavailable",
            "future_same_question_attempts": len(future_same_question),
        }
        rows.append(
            make_row(
                dataset="xes3g5m",
                split=dataset_split,
                task="predict_error_persistence",
                student_id=student_id,
                position=idx,
                history=history,
                query=event,
                target_label=label,
                candidate_targets=persistence,
                metadata=base_metadata,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/unified_student_c1/data/raw/xes3g5m/extracted/XES3G5M"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/xes3g5m_t1_t6"))
    parser.add_argument("--history-k", type=int, default=12)
    parser.add_argument("--future-n", type=int, default=5)
    parser.add_argument("--max-train-per-student", type=int, default=8)
    parser.add_argument("--max-eval-per-student", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = load_questions(args.raw_dir / "metadata")
    split_files = {
        "train": args.raw_dir / "question_level" / "train_valid_sequences_quelevel.csv",
        "test": args.raw_dir / "question_level" / "test_quelevel.csv",
    }
    traces_by_split = {split: read_sequences(path, questions) for split, path in split_files.items()}
    train_students = sorted(traces_by_split["train"])
    rng = random.Random(args.seed)
    rng.shuffle(train_students)
    valid_count = max(1, int(0.1 * len(train_students)))
    valid_ids = set(train_students[:valid_count])
    train_ids = set(train_students[valid_count:])
    assignments = {
        "train": (traces_by_split["train"], train_ids),
        "valid": (traces_by_split["train"], valid_ids),
        "test": (traces_by_split["test"], set(traces_by_split["test"])),
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in assignments}
    for split, (traces, student_ids) in assignments.items():
        maximum = args.max_train_per_student if split == "train" else args.max_eval_per_student
        for student_id in sorted(student_ids):
            trace = traces[student_id]
            positions = choose_positions(trace, maximum, rng)
            rows_by_split[split].extend(make_rows(split, student_id, trace, positions, args.history_k, args.future_n))
    stats = summarize(rows_by_split)
    stats.update({
        "source_dir": str(args.raw_dir),
        "history_k": args.history_k,
        "future_n": args.future_n,
        "students_by_split": {split: len(student_ids) for split, (_, student_ids) in assignments.items()},
        "task_scope": {
            "unavailable": ["predict_response", "predict_prospective_error"],
            "reason": "XES3G5M records correctness but not the student's selected response",
            "predict_error_persistence": "same-question incorrect recurrence, not exact wrong-response recurrence",
        },
    })
    write_dataset(args.out_dir, rows_by_split, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
