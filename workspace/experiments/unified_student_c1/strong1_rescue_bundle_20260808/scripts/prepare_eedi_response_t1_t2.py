#!/usr/bin/env python3
"""Prepare OCR-backed Eedi responses for non-temporal held-out T1/T2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from student_task_utils import make_row, meaningful_text_length, output_tag, split_students, summarize, write_dataset


LETTERS = ["A", "B", "C", "D"]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_ocr(path: Path) -> dict[str, dict[str, Any]]:
    questions = {}
    for row in read_jsonl(path):
        options = row.get("options_ocr") or {}
        question_text = str(row.get("question_text_ocr") or "")
        if row.get("parse_status") != "ok" or meaningful_text_length(question_text) < 4:
            continue
        if any(not str(options.get(letter, "")).strip() for letter in LETTERS):
            continue
        option_text = " ".join(str(options[letter]) for letter in LETTERS)
        if re.search(r"\b(?:which|what|where|when|who|how)\b[^?]{0,80}\?", option_text, flags=re.I):
            continue
        packed_options = re.sub(r"[^a-z]", "", option_text.lower())
        if any(
            cue in packed_options
            for cue in ("whichofthefollowing", "whoiscorrect", "whatisthe", "whatnumber", "howmany", "howwould")
        ):
            continue
        questions[str(row["question_id"])] = row
    return questions


def load_interactions(raw_dir: Path, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped: Counter[str] = Counter()
    for metadata_path in sorted(raw_dir.glob("*/metadata.jsonl")):
        for row_number, row in enumerate(read_jsonl(metadata_path)):
            question_id = str(row["QuestionId"])
            question = questions.get(question_id)
            if question is None:
                dropped["question_failed_ocr_filter"] += 1
                continue
            response = str(row.get("AnswerValue") or "")
            correct_response = str(row.get("CorrectAnswer") or "")
            if response not in LETTERS or correct_response not in LETTERS:
                dropped["invalid_response"] += 1
                continue
            event = {
                "source_row": row_number,
                "question_id": question_id,
                "question_text": str(question["question_text_ocr"]),
                "kc_ids": [str(subject) for subject in row.get("Subjects") or []],
                "kc_names": [str(subject) for subject in row.get("Subjects") or []],
                "options": {letter: str(question["options_ocr"][letter]) for letter in LETTERS},
                "student_response": response,
                "correct_response": correct_response,
                "correctness": int(row["IsCorrect"]),
                "macro_area": str(row.get("MacroArea") or ""),
                "image_path": str(question.get("image_path") or ""),
            }
            traces[str(row["UserId"])].append(event)
    return {student_id: events for student_id, events in traces.items() if len(events) >= 3}, dropped


def stable_order(events: list[dict[str, Any]], student_id: str, seed: int) -> list[dict[str, Any]]:
    def key(event: dict[str, Any]) -> str:
        value = f"{seed}:{student_id}:{event['question_id']}:{event['source_row']}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return sorted(events, key=key)


def target_positions(events: list[dict[str, Any]], maximum: int, rng: random.Random) -> list[int]:
    positions = list(range(len(events)))
    correct = [idx for idx in positions if events[idx]["correctness"]]
    incorrect = [idx for idx in positions if not events[idx]["correctness"]]
    rng.shuffle(correct)
    rng.shuffle(incorrect)
    incorrect_n = min(len(incorrect), (maximum + 1) // 2)
    selected = incorrect[:incorrect_n] + correct[: maximum - incorrect_n]
    if len(selected) < maximum:
        used = set(selected)
        remaining = [idx for idx in positions if idx not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[: maximum - len(selected)])
    return sorted(selected)


def make_rows(
    split: str,
    student_id: str,
    events: list[dict[str, Any]],
    history_k: int,
    maximum: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = []
    binary = {label: output_tag("predict_correctness", label) for label in ("correct", "incorrect")}
    response_targets = {letter: output_tag("predict_response", letter) for letter in LETTERS}
    for idx in target_positions(events, maximum, rng):
        event = events[idx]
        history_pool = [candidate for candidate_idx, candidate in enumerate(events) if candidate_idx != idx]
        history = history_pool[:history_k]
        metadata = {
            "source": "Eedi response Hugging Face mirror",
            "question_id": event["question_id"],
            "skill_id": "|".join(event["kc_ids"]),
            "skill_name": "|".join(event["kc_names"]),
            "target_correctness": event["correctness"],
            "target_response": event["student_response"],
            "sequence_semantics": "non-temporal user-profile held-out response",
            "ordering_limitation": "mirror omits timestamp and AnswerId; deterministic hash order is used only for reproducibility",
            "image_path": event["image_path"],
        }
        common = dict(
            dataset="eedi_response_ocr",
            split=split,
            student_id=student_id,
            position=idx,
            history=history,
            query=event,
            metadata=metadata,
        )
        correctness = "correct" if event["correctness"] else "incorrect"
        rows.append(make_row(task="predict_correctness", target_label=correctness, candidate_targets=binary, **common))
        rows.append(make_row(task="predict_response", target_label=event["student_response"], candidate_targets=response_targets, **common))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/unified_student_c1/data/raw/eedi_response_hf"))
    parser.add_argument("--ocr-jsonl", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_response_ocr/questions_ocr.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_response_t1_t2"))
    parser.add_argument("--history-k", type=int, default=12)
    parser.add_argument("--max-train-per-student", type=int, default=8)
    parser.add_argument("--max-eval-per-student", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = load_ocr(args.ocr_jsonl)
    traces, dropped = load_interactions(args.raw_dir, questions)
    traces = {student_id: stable_order(events, student_id, args.seed) for student_id, events in traces.items()}
    splits = split_students(sorted(traces), args.seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    rng = random.Random(args.seed)
    for split, students in splits.items():
        maximum = args.max_train_per_student if split == "train" else args.max_eval_per_student
        for student_id in sorted(students):
            rows_by_split[split].extend(make_rows(split, student_id, traces[student_id], args.history_k, maximum, rng))
    stats = summarize(rows_by_split)
    stats.update({
        "source_dir": str(args.raw_dir),
        "ocr_jsonl": str(args.ocr_jsonl),
        "ocr_questions_retained": len(questions),
        "dropped_interactions": dict(dropped),
        "history_k": args.history_k,
        "students_by_split": {split: len(students) for split, students in splits.items()},
        "task_scope": {
            "available": ["predict_correctness", "predict_response"],
            "semantics": "non-temporal held-out response conditioned on other responses from the same user",
            "unavailable": ["estimate_mastery", "predict_prospective_error", "predict_error_persistence"],
            "reason": "the public mirror omits timestamps and AnswerId",
        },
    })
    write_dataset(args.out_dir, rows_by_split, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
