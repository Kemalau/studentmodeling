#!/usr/bin/env python3
"""Materialize leak-audited public T1--T5 task-only Qwen training files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_TRAIN = {
    "predict_correctness": 110525,
    "predict_response": 55569,
    "diagnose_misconception": 6776,
    "predict_prospective_error": 78600,
    "predict_error_persistence": 8375,
}
EXPECTED_TEST = {
    "predict_correctness": 10913,
    "predict_response": 3693,
    "diagnose_misconception": 1494,
    "predict_prospective_error": 4600,
    "predict_error_persistence": 976,
}


def source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("eval_source") or metadata.get("mix_source") or row.get("dataset") or "unknown")


def original_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("original_example_id") or row.get("example_id") or "")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(train_path: Path, test_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        task: (output_dir / f"{task}.jsonl").open("w", encoding="utf-8")
        for task in EXPECTED_TRAIN
    }
    train_counts = Counter()
    train_sources = Counter()
    train_splits = Counter()
    train_ids = {task: set() for task in EXPECTED_TRAIN}
    train_original = {task: set() for task in EXPECTED_TRAIN}
    try:
        with train_path.open(encoding="utf-8") as input_handle:
            for line in input_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task = str(row.get("task") or "")
                if task not in handles:
                    raise ValueError(f"Unexpected unified-train task: {task}")
                example_id = str(row.get("example_id") or "")
                if not example_id or example_id in train_ids[task]:
                    raise ValueError(f"Missing or duplicate train example_id: {example_id!r}")
                train_ids[task].add(example_id)
                train_original[task].add(original_id(row))
                train_counts[task] += 1
                train_sources[(task, source(row))] += 1
                train_splits[(task, str((row.get("metadata") or {}).get("original_split") or "unknown"))] += 1
                handles[task].write(line if line.endswith("\n") else line + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    if dict(train_counts) != EXPECTED_TRAIN:
        raise ValueError(f"Train task count drift: {dict(train_counts)}")

    test_counts = Counter()
    test_sources = Counter()
    overlap = Counter()
    test_ids = {task: set() for task in EXPECTED_TEST}
    with test_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row.get("task") or "")
            if task not in EXPECTED_TEST:
                raise ValueError(f"Unexpected formal-test task: {task}")
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in test_ids[task]:
                raise ValueError(f"Missing or duplicate test example_id: {example_id!r}")
            test_ids[task].add(example_id)
            test_counts[task] += 1
            test_sources[(task, source(row))] += 1
            overlap[task] += int(original_id(row) in train_original[task])
    if dict(test_counts) != EXPECTED_TEST:
        raise ValueError(f"Test task count drift: {dict(test_counts)}")
    if any(overlap.values()):
        raise ValueError(f"Train/test original-example leakage: {dict(overlap)}")

    tasks = {}
    for task in EXPECTED_TRAIN:
        task_path = output_dir / f"{task}.jsonl"
        tasks[task] = {
            "train_rows": train_counts[task],
            "test_rows": test_counts[task],
            "train_original_ids": len(train_original[task]),
            "train_test_original_id_overlap": overlap[task],
            "train_sources": {
                source_name: count
                for (task_name, source_name), count in sorted(train_sources.items())
                if task_name == task
            },
            "train_original_splits": {
                split: count
                for (task_name, split), count in sorted(train_splits.items())
                if task_name == task
            },
            "test_sources": {
                source_name: count
                for (task_name, source_name), count in sorted(test_sources.items())
                if task_name == task
            },
            "train_jsonl": task_path.name,
            "train_jsonl_sha256": file_digest(task_path),
        }
    receipt = {
        "stage": "taskonly_qwen_public_t1_t5_data_v1",
        "status": "complete",
        "source_train_jsonl": str(train_path),
        "source_train_sha256": file_digest(train_path),
        "formal_test_jsonl": str(test_path),
        "formal_test_sha256": file_digest(test_path),
        "test_used_for_training": False,
        "tasks": tasks,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.train_jsonl, args.test_jsonl, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
