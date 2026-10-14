#!/usr/bin/env python3
"""Paired headline comparisons for task-only versus unified Qwen predictions."""

from __future__ import annotations

import argparse
import html
import hashlib
import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

TASKS = {
    "T1": ("predict_correctness", 10_913),
    "T2": ("predict_response", 3_693),
    "T3": ("diagnose_misconception", 1_494),
    "T4": ("predict_prospective_error", 4_600),
    "T5": ("predict_error_persistence", 976),
}


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.I | re.S)
    return html.unescape(match.group(1).strip()) if match else None


def normalize_text(value: str | None) -> str:
    text = html.unescape(str(value or "")).casefold().strip()
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    return re.sub(r"\s+", "", text)


def numeric_value(value: str | None):
    text = normalize_text(value).replace(",", "")
    if not text or len(text) > 80 or not re.fullmatch(
        r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+\-]?(?:\d+(?:\.\d+)?|\.\d+))?",
        text,
    ):
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def mathematically_equivalent(prediction: str | None, gold: str | None) -> bool:
    if normalize_text(prediction) == normalize_text(gold):
        return True
    predicted = numeric_value(prediction)
    target = numeric_value(gold)
    return predicted is not None and target is not None and predicted == target


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index(rows: list[dict[str, Any]], task: str, expected: int) -> dict[str, dict[str, Any]]:
    rows = [row for row in rows if row.get("task") == task]
    output = {str(row["example_id"]): row for row in rows}
    if len(rows) != expected or len(output) != expected:
        raise ValueError(f"{task} support/duplicate drift: rows={len(rows)} ids={len(output)} expected={expected}")
    return output


def correctness(task: str) -> Callable[[dict[str, Any]], bool]:
    if task in {"predict_correctness", "predict_error_persistence"}:
        return lambda row: str(row.get("prediction")) == str(row.get("gold"))
    if task == "diagnose_misconception":
        def judged(row: dict[str, Any]) -> bool:
            if "llm_judge_correct" not in row:
                raise ValueError("T3 row lacks an independent semantic judgment")
            return bool(row["llm_judge_correct"])
        return judged
    if task == "predict_response":
        def response(row: dict[str, Any]) -> bool:
            predicted = extract_tag(str(row.get("raw_output") or ""), "predicted_response")
            gold = extract_tag(str(row.get("target") or ""), "predicted_response")
            if predicted is None or gold is None:
                raise ValueError(
                    "T2 row lacks unredacted <predicted_response> fields; "
                    "predictions_redacted.jsonl cannot be used for paired scoring"
                )
            return mathematically_equivalent(predicted, gold)
        return response
    if task == "predict_prospective_error":
        def joint(row: dict[str, Any]) -> bool:
            if "is_correct" in row:
                if not isinstance(row["is_correct"], bool):
                    raise ValueError("T4 is_correct must be Boolean")
                return row["is_correct"]
            predicted_correctness = extract_tag(str(row.get("raw_output") or ""), "correctness")
            gold_correctness = extract_tag(str(row.get("target") or ""), "correctness")
            predicted_response = extract_tag(str(row.get("raw_output") or ""), "predicted_response")
            gold_response = extract_tag(str(row.get("target") or ""), "predicted_response")
            if any(value is None for value in (
                predicted_correctness,
                gold_correctness,
                predicted_response,
                gold_response,
            )):
                raise ValueError(
                    "T4 row lacks unredacted correctness/response fields; "
                    "predictions_redacted.jsonl cannot be used for paired scoring"
                )
            return (
                normalize_text(predicted_correctness) == normalize_text(gold_correctness)
                and mathematically_equivalent(predicted_response, gold_response)
            )
        return joint
    raise ValueError(f"Unsupported task: {task}")


def binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    observed = math.comb(trials, successes) / (2**trials)
    return min(
        1.0,
        sum(
            math.comb(trials, value) / (2**trials)
            for value in range(trials + 1)
            if math.comb(trials, value) / (2**trials) <= observed + 1e-15
        ),
    )


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position)); high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def compare(
    unified_rows: list[dict[str, Any]],
    taskonly_rows: list[dict[str, Any]],
    public_task: str,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    task, expected = TASKS[public_task]
    unified = index(unified_rows, task, expected)
    taskonly = index(taskonly_rows, task, expected)
    if set(unified) != set(taskonly):
        raise ValueError(f"{public_task} example-ID sets differ")
    score = correctness(task)
    ids = sorted(unified)
    u = [score(unified[key]) for key in ids]
    s = [score(taskonly[key]) for key in ids]
    taskonly_only = sum(sv and not uv for uv, sv in zip(u, s))
    unified_only = sum(uv and not sv for uv, sv in zip(u, s))
    both_correct = sum(uv and sv for uv, sv in zip(u, s))
    both_wrong = expected - taskonly_only - unified_only - both_correct
    differences = [float(sv) - float(uv) for uv, sv in zip(u, s)]
    rng = random.Random(seed)
    boot = [
        sum(differences[rng.randrange(expected)] for _ in range(expected)) / expected
        for _ in range(bootstrap_samples)
    ]
    return {
        "public_task": public_task,
        "internal_task": task,
        "n": expected,
        "taskonly_primary": sum(s) / expected,
        "unified_primary": sum(u) / expected,
        "taskonly_minus_unified": sum(differences) / expected,
        "paired_bootstrap_95ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "discordant": {
            "taskonly_only_correct": taskonly_only,
            "unified_only_correct": unified_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "mcnemar_exact_two_sided_p": binomial_two_sided(taskonly_only, taskonly_only + unified_only),
        "example_id_set_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--unified", type=Path, nargs="+", required=True)
    parser.add_argument("--taskonly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    unified = [row for path in args.unified for row in read_jsonl(path)]
    taskonly = read_jsonl(args.taskonly)
    value = compare(unified, taskonly, args.task, args.bootstrap_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
