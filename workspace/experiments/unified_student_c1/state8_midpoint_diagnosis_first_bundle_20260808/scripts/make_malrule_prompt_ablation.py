#!/usr/bin/env python3
"""Build prompt ablations for Malrule C1 diagnostics."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


STATE_TOKEN = "<STUDENT_STATE>"


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
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def history_segment(history: list[dict[str, Any]], corrected: bool = False) -> str:
    lines: list[str] = []
    for idx, event in enumerate(history, 1):
        answer = event["correct_answer"] if corrected else event["student_answer"]
        is_correct = True if corrected else bool(event.get("is_correct"))
        lines.extend(
            [
                f"Trial {idx}:",
                f"Problem: {event['problem_text']}",
                f"Student answer: {answer}",
                f"Correct answer: {event['correct_answer']}",
                f"Outcome: {'correct' if is_correct else 'incorrect'}",
                "",
            ]
        )
    body = "\n".join(lines).rstrip()
    return f"<student_history>\n{body}\n</student_history>\n" if body else "<student_history>\n</student_history>\n"


def question_segment(row: dict[str, Any], reverse_candidates: bool = False) -> str:
    query = row["query"]
    labels = list(row["candidate_labels"])
    if reverse_candidates:
        labels = list(reversed(labels))
    lines = [
        f"Problem: {query['problem_text']}",
        f"Correct answer: {query['correct_answer']}",
        "Valid outputs:",
    ]
    for label in labels:
        lines.append(row["candidate_targets"][label])
    return "<query>\n" + "\n".join(lines) + "\n</query>\nAnswer:"


def rebuild_prompt(row: dict[str, Any]) -> None:
    row["prompt"] = f"{row['seg_instruction']}{row['seg_history']}{STATE_TOKEN}\n{row['seg_question']}"


def make_wrong_history_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    donors: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("metadata") or {}
        donors[(row["task"], meta.get("family", ""))].append(row)

    out = []
    for row in rows:
        new = dict(row)
        meta = row.get("metadata") or {}
        pool = [
            cand
            for cand in donors[(row["task"], meta.get("family", ""))]
            if (cand.get("metadata") or {}).get("malrule_id") != meta.get("malrule_id")
        ]
        if not pool:
            pool = [cand for cand in rows if cand["task"] == row["task"] and cand["example_id"] != row["example_id"]]
        donor = rng.choice(pool)
        new["history"] = donor["history"]
        new["seg_history"] = donor["seg_history"]
        rebuild_prompt(new)
        out.append(new)
    return out


def ablate(rows: list[dict[str, Any]], condition: str, seed: int) -> list[dict[str, Any]]:
    if condition == "original":
        return [dict(row) for row in rows]
    if condition == "wrong_history_same_family":
        return make_wrong_history_rows(rows, seed)

    rng = random.Random(seed)
    out = []
    for row in rows:
        new = dict(row)
        if condition == "no_history":
            new["history"] = []
            new["seg_history"] = history_segment([])
        elif condition == "corrected_history":
            new["seg_history"] = history_segment(row["history"], corrected=True)
        elif condition == "shuffled_history":
            hist = list(row["history"])
            rng.shuffle(hist)
            new["history"] = hist
            new["seg_history"] = history_segment(hist)
        elif condition == "reversed_candidates":
            new["candidate_labels"] = list(reversed(row["candidate_labels"]))
            new["candidate_targets"] = {label: row["candidate_targets"][label] for label in new["candidate_labels"]}
            new["seg_question"] = question_segment(row, reverse_candidates=True)
        else:
            raise ValueError(condition)
        rebuild_prompt(new)
        out.append(new)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["malrule_identity", "malrule_next_response"],
    )
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.input_jsonl) if row["task"] in set(args.tasks)]
    for condition in [
        "original",
        "no_history",
        "wrong_history_same_family",
        "corrected_history",
        "shuffled_history",
        "reversed_candidates",
    ]:
        write_jsonl(args.out_dir / f"{condition}.jsonl", ablate(rows, condition, args.seed))


if __name__ == "__main__":
    main()
