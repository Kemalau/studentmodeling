#!/usr/bin/env python3
"""Prepare compact train/valid/test files for formal specialist baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TASK_ALIASES = {
    "malrule_next_correctness": "predict_correctness",
    "malrule_next_response": "predict_response",
    "malrule_identity": "diagnose_misconception",
}


def canonical_task(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "")
    return TASK_ALIASES.get(task, task)


def source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("eval_source") or metadata.get("mix_source") or row.get("dataset") or "unknown")


def canonical_student_id(row: dict[str, Any], source_name: str) -> str:
    student_id = str(row.get("student_id") or "unknown")
    prefix = f"{source_name}::"
    return student_id if student_id.startswith(prefix) else prefix + student_id


def stable_valid(row: dict[str, Any], source_name: str, valid_percent: int) -> bool:
    task = canonical_task(row)
    if task == "diagnose_misconception":
        value = f"{source_name}::{task}::{original_id(row)}"
    else:
        value = f"{canonical_student_id(row, source_name)}::{task}"
    bucket = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < valid_percent


def original_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("original_example_id") or row.get("example_id") or "")


def compact(row: dict[str, Any], source_name: str, task: str, split: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("mix_source", source_name)
    metadata.setdefault("question_id", (row.get("query") or {}).get("question_id") or (row.get("query") or {}).get("surface_id") or "")
    if not metadata.get("skill_id"):
        query_kcs = (row.get("query") or {}).get("kc_ids") or []
        metadata["skill_id"] = "|".join(str(value) for value in query_kcs)
    return {
        "example_id": str(row.get("example_id") or original_id(row)),
        "dataset": source_name,
        "split": split,
        "student_id": canonical_student_id(row, source_name),
        "task": task,
        "history": row.get("history") or [],
        "query": row.get("query") or {},
        "candidate_labels": row.get("candidate_labels") or [],
        "candidate_targets": row.get("candidate_targets") or {},
        "target_label": row.get("target_label"),
        "target": row.get("target"),
        "metadata": metadata,
    }


def formal_pairs(test_jsonl: Path) -> set[tuple[str, str]]:
    pairs = set()
    with test_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                pairs.add((canonical_task(row), source(row)))
    return pairs


def write_test(test_jsonl: Path, output_path: Path, pairs: set[tuple[str, str]], counts: Counter[str]) -> None:
    with test_jsonl.open("r", encoding="utf-8") as source_handle, output_path.open("w", encoding="utf-8") as output_handle:
        for line in source_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = canonical_task(row)
            source_name = source(row)
            if (task, source_name) not in pairs:
                continue
            output_handle.write(json.dumps(compact(row, source_name, task, "test"), ensure_ascii=False, sort_keys=True) + "\n")
            counts[f"test/{task}/{source_name}"] += 1


def write_train_valid(
    train_jsonl: Path,
    train_path: Path,
    valid_path: Path,
    pairs: set[tuple[str, str]],
    valid_percent: int,
    counts: Counter[str],
) -> None:
    seen: set[tuple[str, str, str]] = set()
    with train_jsonl.open("r", encoding="utf-8") as source_handle, train_path.open("w", encoding="utf-8") as train_handle, valid_path.open("w", encoding="utf-8") as valid_handle:
        for line in source_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = canonical_task(row)
            source_name = source(row)
            if (task, source_name) not in pairs:
                continue
            key = (task, source_name, original_id(row))
            if key in seen:
                counts["duplicates_dropped"] += 1
                continue
            seen.add(key)
            is_valid = stable_valid(row, source_name, valid_percent)
            split = "valid" if is_valid else "train"
            target_handle = valid_handle if is_valid else train_handle
            target_handle.write(json.dumps(compact(row, source_name, task, split), ensure_ascii=False, sort_keys=True) + "\n")
            counts[f"{split}/{task}/{source_name}"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-percent", type=int, default=10)
    args = parser.parse_args()

    if not 1 <= args.valid_percent <= 50:
        raise ValueError("valid-percent must be between 1 and 50")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = formal_pairs(args.formal_test_jsonl)
    counts: Counter[str] = Counter()
    write_train_valid(
        args.train_jsonl,
        args.output_dir / "train.jsonl",
        args.output_dir / "valid.jsonl",
        pairs,
        args.valid_percent,
        counts,
    )
    write_test(args.formal_test_jsonl, args.output_dir / "test.jsonl", pairs, counts)
    stats = {
        "train_jsonl": str(args.train_jsonl),
        "formal_test_jsonl": str(args.formal_test_jsonl),
        "valid_percent": args.valid_percent,
        "formal_task_source_pairs": [list(pair) for pair in sorted(pairs)],
        "counts": dict(sorted(counts.items())),
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
