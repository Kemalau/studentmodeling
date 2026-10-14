#!/usr/bin/env python3
"""Prepare privacy-filtered FoundationalASSIST student-modeling tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from student_task_utils import clean_text, split_students, write_dataset


STATE_TOKEN = "<STUDENT_STATE>"
STRUCTURED_ANSWER_TYPES = {
    "Numeric",
    "Algebraic Expression",
    "Exact Fraction",
    "Numeric Expression",
    "Ordering",
    "Drop Down",
    "Multiple Choice",
    "Check All That Apply",
}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)")
LONG_WORD_RE = re.compile(r"[A-Za-z]{12,}")
MATH_WORDS = {"pi", "sqrt", "sin", "cos", "tan", "log", "ln"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_response(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def safe_response(value: Any, answer_type: str, authored_options: list[str]) -> tuple[str, str]:
    response = normalize_response(value)
    if not response:
        return "", "empty"
    if len(response) > 128:
        return "", "too_long"
    if EMAIL_RE.search(response) or URL_RE.search(response) or PHONE_RE.search(response):
        return "", "pii_pattern"
    option_lookup = {normalize_response(option).casefold(): normalize_response(option) for option in authored_options}
    if response.casefold() in option_lookup:
        return option_lookup[response.casefold()], "authored_option"
    if answer_type not in STRUCTURED_ANSWER_TYPES:
        return "", "unstructured_answer_type"
    if LONG_WORD_RE.search(response):
        return "", "long_alpha_token"
    alpha_tokens = re.findall(r"[A-Za-z]+", response)
    has_math_signal = bool(re.search(r"[0-9+\-*/^=()\[\]{}<>.%]", response))
    safe_symbolic_words = all(len(token) == 1 or token.casefold() in MATH_WORDS for token in alpha_tokens)
    if alpha_tokens and not has_math_signal and not safe_symbolic_words:
        return "", "free_text"
    return response, "structured_response"


def option_values(row: dict[str, str]) -> list[str]:
    values = []
    for field, separator in (("Multiple Choice Options", "||"), ("Fill-in Options", "</p>,")):
        raw = str(row.get(field) or "").strip()
        if not raw:
            continue
        values.extend(clean_text(value) for value in raw.split(separator) if clean_text(value))
    return list(dict.fromkeys(values))


def correct_values(row: dict[str, str]) -> list[str]:
    values = []
    for field, separator in (("Multiple Choice Answers", "||"), ("Fill-in Answers", ",")):
        raw = str(row.get(field) or "").strip()
        if not raw:
            continue
        values.extend(clean_text(value) for value in raw.split(separator) if clean_text(value))
    return list(dict.fromkeys(values))


def load_problems(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    payloads: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        payload = {key: str(value or "") for key, value in row.items()}
        signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        grouped[str(row["problem_id"])][signature] += 1
        payloads[signature] = payload
    problems = {}
    conflicting = 0
    for problem_id, signatures in grouped.items():
        if len(signatures) > 1:
            conflicting += 1
        signature = max(signatures, key=lambda value: (signatures[value], value))
        row = payloads[signature]
        options = option_values(row)
        correct = correct_values(row)
        problems[problem_id] = {
            "question_text": clean_text(row.get("Problem Body")),
            "question_type": clean_text(row.get("Problem Type")),
            "answer_type": clean_text(row.get("Answer Types")),
            "options": options,
            "correct_responses": correct,
            "problem_set_id": str(row.get("Problem Set Id") or ""),
            "problem_part": str(row.get("Problem Part") or ""),
        }
    return problems, {
        "problem_rows": sum(sum(counter.values()) for counter in grouped.values()),
        "unique_problems": len(problems),
        "conflicting_problem_ids": conflicting,
    }


def load_skills(path: Path) -> dict[str, list[dict[str, str]]]:
    skills: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        item = {
            "skill_id": str(row["skill_id"]),
            "skill_code": clean_text(row.get("node_code")),
            "skill_name": clean_text(row.get("node_name")),
        }
        if item not in skills[str(row["problem_id"])]:
            skills[str(row["problem_id"])].append(item)
    return skills


def load_traces(
    path: Path,
    problems: dict[str, dict[str, Any]],
    skills: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids = set()
    stats = Counter()
    privacy = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stats["raw_rows"] += 1
            interaction_id = str(row["id"])
            if interaction_id in seen_ids:
                stats["duplicate_rows_removed"] += 1
                continue
            seen_ids.add(interaction_id)
            problem_id = str(row["problem_id"])
            problem = problems.get(problem_id)
            if problem is None:
                stats["missing_problem"] += 1
                continue
            timestamp = str(row.get("end_time") or "").strip()
            if not timestamp:
                stats["missing_timestamp"] += 1
                continue
            try:
                score_value = float(row.get("discrete_score") or "nan")
            except ValueError:
                stats["invalid_score"] += 1
                continue
            if score_value not in {0.0, 1.0}:
                stats["invalid_score"] += 1
                continue
            response, reason = safe_response(row.get("answer_text"), problem["answer_type"], problem["options"])
            privacy[reason] += 1
            if not response:
                stats["unsafe_or_missing_response"] += 1
                continue
            event_skills = skills.get(problem_id, [])
            event = {
                "interaction_id": interaction_id,
                "timestamp": timestamp,
                "question_id": problem_id,
                "question_text": problem["question_text"],
                "question_type": problem["question_type"],
                "answer_type": problem["answer_type"],
                "problem_set_id": problem["problem_set_id"],
                "problem_part": problem["problem_part"],
                "kc_ids": [item["skill_id"] for item in event_skills],
                "kc_names": [item["skill_name"] for item in event_skills],
                "options": {str(index + 1): value for index, value in enumerate(problem["options"])},
                "student_response": response,
                "correct_response": " || ".join(problem["correct_responses"]),
                "hint_count": int(float(row.get("hint_count") or 0)),
                "saw_answer": str(row.get("saw_answer") or "").strip().lower() == "true",
            }
            event["unsupported_first_attempt_success"] = int(score_value)
            event["first_response_correctness"] = (
                1 if score_value == 1.0
                else 0 if not event["hint_count"] and not event["saw_answer"]
                else None
            )
            traces[str(row["user_id"])].append(event)
            stats["kept_rows"] += 1
    for trace in traces.values():
        trace.sort(key=lambda event: (event["timestamp"], int(event["interaction_id"])))
    traces = {student_id: trace for student_id, trace in traces.items() if len(trace) >= 6}
    stats["students"] = len(traces)
    stats["privacy_reasons"] = dict(privacy)
    return traces, dict(stats)


def xml_value(value: str) -> str:
    return html.escape(value, quote=False)


def response_target(response: str) -> str:
    return f"<predicted_response>{xml_value(response)}</predicted_response>"


def prospective_target(event: dict[str, Any]) -> str:
    correctness = "correct" if event["first_response_correctness"] else "incorrect"
    return f"<correctness>{correctness}</correctness>{response_target(event['student_response'])}"


def format_event(event: dict[str, Any], number: int) -> str:
    correctness = event["first_response_correctness"]
    if correctness is None:
        outcome = "unknown because support was used"
    else:
        outcome = "correct" if correctness else "incorrect"
    lines = [
        f"Trial {number}:",
        f"Knowledge components: {', '.join(event['kc_names'])}",
        f"Question: {event['question_text']}",
        f"Student response: {xml_value(event['student_response'])}",
        f"Correct response: {xml_value(event['correct_response'])}",
        f"First-response outcome: {outcome}",
    ]
    return "\n".join(lines)


def format_query(event: dict[str, Any], include_observed_response: bool = False) -> str:
    lines = [
        f"Knowledge components: {', '.join(event['kc_names'])}",
        f"Question: {event['question_text']}",
    ]
    if event["options"]:
        lines.append("Answer choices:")
        lines.extend(f"- {key}: {value}" for key, value in event["options"].items())
    if include_observed_response:
        lines.append(f"Observed error response: {xml_value(event['student_response'])}")
    return "\n".join(lines)


def build_prompt(
    task: str,
    history: list[dict[str, Any]],
    query: dict[str, Any],
    prospective_delta: int | None = None,
) -> str:
    instructions = {
        "predict_correctness": "Predict whether the student will answer the queried question correctly.",
        "predict_response": "Generate the exact first response the student will give to the queried question.",
        "predict_error_persistence": "Predict whether the same exact wrong response will recur within the next three attempts on this question.",
    }
    if task == "predict_prospective_error":
        instruction = (
            f"The queried question occurs {prospective_delta} interactions after the visible history. "
            "Predict both first-attempt success and the student's exact first response; intervening interactions are hidden."
        )
    else:
        instruction = instructions[task]
    history_text = "\n\n".join(format_event(event, index) for index, event in enumerate(history, 1))
    query_text = format_query(query, include_observed_response=task == "predict_error_persistence")
    if task == "predict_correctness":
        output_rule = "Valid outputs:\n<correctness>correct</correctness>\n<correctness>incorrect</correctness>"
    elif task == "predict_error_persistence":
        output_rule = "Valid outputs:\n<persistence>persistent</persistence>\n<persistence>transient</persistence>"
    elif task == "predict_response":
        output_rule = "Output format: <predicted_response>student's exact response</predicted_response>"
    else:
        output_rule = (
            "Output format: <correctness>correct or incorrect</correctness>"
            "<predicted_response>student's exact response</predicted_response>"
        )
    return (
        "You are a student modeling system.\n"
        f"<task>{task}</task>\n"
        f"<instruction>{instruction} Return exactly one valid output.</instruction>\n"
        "<student_history>\n"
        f"{history_text}\n"
        "</student_history>\n"
        f"{STATE_TOKEN}\n"
        "<query>\n"
        f"{query_text}\n"
        "</query>\n"
        f"{output_rule}\n"
        "Answer:"
    )


def metadata(event: dict[str, Any], **extra: Any) -> dict[str, Any]:
    result = {
        "source": "FoundationalASSIST",
        "question_id": event["question_id"],
        "skill_id": "|".join(event["kc_ids"]),
        "skill_name": "|".join(event["kc_names"]),
        "timestamp": event["timestamp"],
        "answer_type": event["answer_type"],
        "problem_set_id": event["problem_set_id"],
        "problem_part": event["problem_part"],
        "score_semantics": "first-response correctness is observed only for score=1 or score=0 without hint/answer support",
    }
    result.update(extra)
    return result


def make_row(
    *,
    split: str,
    task: str,
    student_id: str,
    position: int,
    history: list[dict[str, Any]],
    query: dict[str, Any],
    target_label: str,
    target: str,
    candidate_targets: dict[str, str],
    prospective_delta: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_view = {
        key: value for key, value in query.items()
        if key not in {
            "student_response",
            "correct_response",
            "first_response_correctness",
            "unsupported_first_attempt_success",
            "hint_count",
            "saw_answer",
        }
    }
    if task == "predict_error_persistence":
        query_view["student_response"] = query["student_response"]
    return {
        "example_id": f"foundationalassist:{split}:{task}:{student_id}:{position}:{prospective_delta or 0}",
        "dataset": "foundationalassist",
        "split": split,
        "task": task,
        "student_id": f"foundationalassist::{student_id}",
        "history": history,
        "query": query_view,
        "target_label": target_label,
        "target": target,
        "candidate_labels": list(candidate_targets),
        "candidate_targets": candidate_targets,
        "prompt": build_prompt(task, history, query, prospective_delta),
        "metadata": metadata(query, **(extra_metadata or {})),
    }


def sample_positions(trace: list[dict[str, Any]], maximum: int, max_delta: int, rng: random.Random) -> list[int]:
    positions = [
        position for position in range(1, max(1, len(trace) - max_delta))
        if trace[position]["first_response_correctness"] is not None
    ]
    if len(positions) <= maximum:
        return positions
    correct = [position for position in positions if trace[position]["first_response_correctness"]]
    incorrect = [position for position in positions if not trace[position]["first_response_correctness"]]
    rng.shuffle(correct)
    rng.shuffle(incorrect)
    half = maximum // 2
    selected = incorrect[:half] + correct[: maximum - half]
    if len(selected) < maximum:
        remaining = [position for position in positions if position not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: maximum - len(selected)])
    return sorted(selected)


def rows_for_student(
    split: str,
    student_id: str,
    trace: list[dict[str, Any]],
    history_k: int,
    maximum: int,
    deltas: list[int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = []
    binary = {
        "correct": "<correctness>correct</correctness>",
        "incorrect": "<correctness>incorrect</correctness>",
    }
    persistence = {
        "persistent": "<persistence>persistent</persistence>",
        "transient": "<persistence>transient</persistence>",
    }
    positions = sample_positions(trace, maximum, max(deltas), rng)
    for position in positions:
        event = trace[position]
        history = trace[max(0, position - history_k):position]
        correctness = "correct" if event["first_response_correctness"] else "incorrect"
        rows.append(make_row(
            split=split,
            task="predict_correctness",
            student_id=student_id,
            position=position,
            history=history,
            query=event,
            target_label=correctness,
            target=binary[correctness],
            candidate_targets=binary,
        ))
        response = event["student_response"]
        rows.append(make_row(
            split=split,
            task="predict_response",
            student_id=student_id,
            position=position,
            history=history,
            query=event,
            target_label=response,
            target=response_target(response),
            candidate_targets={},
            extra_metadata={"evaluation_protocol": "open generation; candidate list must not be used"},
        ))
        visible_history = trace[max(0, position - history_k + 1):position + 1]
        for delta in deltas:
            future_position = position + delta
            if future_position >= len(trace):
                continue
            future = trace[future_position]
            if future["first_response_correctness"] is None:
                continue
            future_response = future["student_response"]
            rows.append(make_row(
                split=split,
                task="predict_prospective_error",
                student_id=student_id,
                position=future_position,
                history=visible_history,
                query=future,
                target_label=future_response,
                target=prospective_target(future),
                candidate_targets={},
                prospective_delta=delta,
                extra_metadata={
                    "prospective_delta": delta,
                    "history_cutoff_position": position,
                    "evaluation_protocol": "open generation; candidate list must not be used",
                },
            ))
    for position, event in enumerate(trace):
        if event["first_response_correctness"] != 0:
            continue
        future_same = [candidate for candidate in trace[position + 1:] if candidate["question_id"] == event["question_id"]][:3]
        if not future_same:
            continue
        normalized = event["student_response"].casefold()
        repeats = any(candidate["student_response"].casefold() == normalized for candidate in future_same)
        label = "persistent" if repeats else "transient"
        history = trace[max(0, position - history_k):position]
        rows.append(make_row(
            split=split,
            task="predict_error_persistence",
            student_id=student_id,
            position=position,
            history=history,
            query=event,
            target_label=label,
            target=persistence[label],
            candidate_targets=persistence,
            extra_metadata={
                "persistence_definition": "same question and exact same normalized wrong first response recurs within next 3 same-question records",
                "future_same_question_attempts": len(future_same),
                "strict_current_error_filter": "discrete_score=0, hint_count=0, saw_answer=false, privacy-safe nonempty first response",
            },
        ))
    return rows


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    open_tasks = ("predict_response", "predict_prospective_error")
    open_target_stats = {}
    for task in open_tasks:
        targets = [row["target_label"] for row in all_rows if row["task"] == task]
        lengths = sorted(len(target) for target in targets)
        open_target_stats[task] = {
            "count": len(targets),
            "unique_normalized_hashes": len({
                hashlib.sha256(target.casefold().encode("utf-8")).hexdigest() for target in targets
            }),
            "length_chars": {
                "min": lengths[0] if lengths else 0,
                "median": lengths[len(lengths) // 2] if lengths else 0,
                "p95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0,
                "max": lengths[-1] if lengths else 0,
            },
        }
    return {
        "rows_by_split": {split: len(rows) for split, rows in rows_by_split.items()},
        "rows_by_task": dict(Counter(row["task"] for row in all_rows)),
        "rows_by_task_split": {
            split: dict(Counter(row["task"] for row in rows)) for split, rows in rows_by_split.items()
        },
        "labels_by_task": {
            task: dict(Counter(row["target_label"] for row in all_rows if row["task"] == task))
            for task in ("predict_correctness", "predict_error_persistence")
        },
        "open_target_stats": open_target_stats,
        "prospective_delta_counts": dict(Counter(
            str(row["metadata"].get("prospective_delta"))
            for row in all_rows if row["task"] == "predict_prospective_error"
        )),
        "students": len({row["student_id"] for row in all_rows}),
        "questions": len({row["metadata"]["question_id"] for row in all_rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("/home/v-yuejiali/FoundationalASSIST/Data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--history-k", type=int, default=12)
    parser.add_argument("--prospective-deltas", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--max-train-per-student", type=int, default=8)
    parser.add_argument("--max-eval-per-student", type=int, default=4)
    parser.add_argument("--max-students", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    problems, problem_stats = load_problems(args.raw_dir / "Problems.csv")
    skills = load_skills(args.raw_dir / "Skills.csv")
    traces, trace_stats = load_traces(args.raw_dir / "Interactions.csv", problems, skills)
    if args.max_students:
        traces = {student_id: traces[student_id] for student_id in sorted(traces)[:args.max_students]}
    splits = split_students(sorted(traces), args.seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    rng = random.Random(args.seed)
    for split, students in splits.items():
        maximum = args.max_train_per_student if split == "train" else args.max_eval_per_student
        for student_id in sorted(students):
            rows_by_split[split].extend(rows_for_student(
                split,
                student_id,
                traces[student_id],
                args.history_k,
                maximum,
                args.prospective_deltas,
                rng,
            ))
    stats = summarize(rows_by_split)
    stats.update({
        "source_dir": str(args.raw_dir),
        "problem_stats": problem_stats,
        "trace_stats": trace_stats,
        "students_by_split": {split: len(students) for split, students in splits.items()},
        "history_k": args.history_k,
        "prospective_deltas": args.prospective_deltas,
        "max_students": args.max_students,
        "privacy_policy": {
            "raw_answer_text_reported": False,
            "excluded_answer_types": sorted(set(row["answer_type"] for row in problems.values()) - STRUCTURED_ANSWER_TYPES),
            "filters": [
                "email",
                "url",
                "US-style phone",
                "length>128",
                "long alphabetic token",
                "non-authored free text except single-letter variables and common math words",
            ],
        },
        "task_scope": {
            "predict_correctness": "first-response correctness where discrete_score semantics make the label identifiable",
            "predict_response": "open generation of privacy-filtered exact first response",
            "predict_prospective_error": "open generation at deltas 1, 3, and 5",
            "predict_error_persistence": "strict exact wrong-response recurrence",
        },
    })
    write_dataset(args.out_dir, rows_by_split, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
