#!/usr/bin/env python3
"""Rebuild processed Eedi C1 JSONL rows with OCR question text/options."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LETTERS = ["A", "B", "C", "D"]
TASKS = ["next_correctness", "next_response", "future_mastery"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_ocr(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in read_jsonl(path)}


def option_text(options: dict[str, str], label: str) -> str:
    text = str(options.get(label) or "").strip()
    return text if text else f"Option {label}"


def answer_with_text(label: str, options: dict[str, str]) -> str:
    label = str(label)
    if label in LETTERS:
        return f"{label} ({option_text(options, label)})"
    return label


def options_block(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {option_text(options, letter)}" for letter in LETTERS)


def apply_ocr_to_trial(trial: dict[str, Any], ocr_by_qid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = dict(trial)
    qid = str(out.get("question_id") or "")
    ocr = ocr_by_qid.get(qid)
    if not ocr:
        out["ocr_status"] = "missing_ocr"
        return out

    question_text = str(ocr.get("question_text_ocr") or "").strip()
    if question_text:
        out["question_text"] = question_text
    else:
        out["question_text"] = str(out.get("question_text") or f"Question {qid}")

    options = ocr.get("options_ocr") or {}
    out["options"] = {letter: option_text(options, letter) for letter in LETTERS}
    out["ocr_status"] = ocr.get("parse_status") or "unknown"
    out["ocr_avg_score"] = ocr.get("avg_score")
    out["image_path"] = ocr.get("image_path")
    return out


def output_tag(task: str, label: str) -> str:
    if task == "next_correctness":
        return f"<correctness>{label}</correctness>"
    if task == "next_response":
        return f"<predicted_response>{label}</predicted_response>"
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
        return "Estimate the student's near-future mastery on the queried skill, using low, medium, or high."
    raise ValueError(task)


def format_history(history: list[dict[str, Any]]) -> str:
    parts = []
    for i, trial in enumerate(history, 1):
        options = trial.get("options") or {}
        parts.append(
            "\n".join(
                [
                    f"Trial {i}:",
                    f"Skill: {trial['skill_name']}",
                    f"Question: {trial['question_text']}",
                    "Options:",
                    options_block(options),
                    f"Student answer: {answer_with_text(trial['student_response'], options)}",
                    f"Correct answer: {answer_with_text(trial['correct_response'], options)}",
                    f"Outcome: {'correct' if trial['correctness'] else 'incorrect'}",
                ]
            )
        )
    return "\n\n".join(parts)


def format_query(query: dict[str, Any]) -> str:
    options = query.get("options") or {}
    return "\n".join(
        [
            f"Skill: {query['skill_name']}",
            f"Question: {query['question_text']}",
            "Options:",
            options_block(options),
            f"Known correct answer: {answer_with_text(query['correct_response'], options)}",
        ]
    )


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


def rebuild_row(row: dict[str, Any], ocr_by_qid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    out["history"] = [apply_ocr_to_trial(trial, ocr_by_qid) for trial in row.get("history", [])]
    out["query"] = apply_ocr_to_trial(row["query"], ocr_by_qid)
    task = out["task"]
    label = str(out["target_label"])
    out["target"] = output_tag(task, label)
    out["candidate_targets"] = {cand: output_tag(task, cand) for cand in labels_for_task(task)}
    out["prompt"] = build_prompt(task, out["history"], out["query"])
    out.setdefault("metadata", {})
    out["metadata"] = dict(out["metadata"])
    out["metadata"]["text_source"] = "rapidocr_eedi_image"
    out["metadata"]["query_ocr_status"] = out["query"].get("ocr_status")
    out["metadata"]["query_image_path"] = out["query"].get("image_path")
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split_task = Counter((row["split"], row["task"]) for row in rows)
    label_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for task in TASKS:
        label_counts[task] = dict(Counter(row["target_label"] for row in rows if row["task"] == task))
    query_status = Counter(row.get("metadata", {}).get("query_ocr_status") for row in rows)
    prompt_lengths = [len(row["prompt"]) for row in rows]
    return {
        "n_examples": len(rows),
        "by_split_task": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_split_task.items())},
        "label_counts": label_counts,
        "students": len({row["student_id"] for row in rows}),
        "questions": len({row["metadata"]["question_id"] for row in rows}),
        "skills": len({row["metadata"]["skill_id"] for row in rows}),
        "query_ocr_status_counts": dict(query_status),
        "prompt_chars_mean": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        "prompt_chars_max": max(prompt_lengths) if prompt_lengths else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_c1_k12"))
    parser.add_argument("--ocr-jsonl", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_response_ocr/questions_ocr.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_c1_k12_ocr"))
    args = parser.parse_args()

    ocr_by_qid = load_ocr(args.ocr_jsonl)
    out_rows: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for split in ["train", "valid", "test"]:
        rows = [rebuild_row(row, ocr_by_qid) for row in read_jsonl(args.source_dir / f"{split}.jsonl")]
        rows.sort(key=lambda r: r["example_id"])
        out_rows[split] = rows
        all_rows.extend(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in out_rows.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", rows)
        for task in TASKS:
            write_jsonl(args.out_dir / f"{split}_{task}.jsonl", [row for row in rows if row["task"] == task])

    stats = summarize(all_rows)
    stats["source_dir"] = str(args.source_dir)
    stats["ocr_jsonl"] = str(args.ocr_jsonl)
    stats["target_schema_note"] = "next_response uses <predicted_response> to match StudentModeling/TaskC1.md"
    (args.out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
