#!/usr/bin/env python3
"""Freeze the five-source T1 train/valid/test rows used by modern KT baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TASK = "predict_correctness"
EXPECTED = {"train": 99_167, "valid": 11_358, "test": 10_913}
EXPECTED_SOURCES = {
    "dbe_kt22",
    "eedi_response",
    "foundationalassist",
    "malrule",
    "xes3g5m",
}


def source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("eval_source") or metadata.get("mix_source") or row.get("dataset") or "unknown")


def original_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("original_example_id") or row.get("example_id") or "")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def materialize(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {}
    split_students: dict[str, set[tuple[str, str]]] = {}
    split_original_ids: dict[str, set[tuple[str, str]]] = {}

    for split in ("train", "valid", "test"):
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"
        counts: Counter[str] = Counter()
        labels: Counter[str] = Counter()
        students: set[tuple[str, str]] = set()
        ids: set[tuple[str, str]] = set()
        rows = 0
        with input_path.open(encoding="utf-8") as source_handle, output_path.open("w", encoding="utf-8") as output_handle:
            for line in source_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("task") != TASK:
                    continue
                source_name = source(row)
                example_key = (source_name, original_id(row))
                if example_key in ids:
                    raise ValueError(f"Duplicate original example in {split}: {example_key}")
                ids.add(example_key)
                students.add((source_name, str(row.get("student_id") or "")))
                counts[source_name] += 1
                labels[f"{source_name}/{row.get('target_label')}"] += 1
                rows += 1
                output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if rows != EXPECTED[split]:
            raise ValueError(f"{split} row drift: expected {EXPECTED[split]}, got {rows}")
        if set(counts) != EXPECTED_SOURCES:
            raise ValueError(f"{split} source drift: {sorted(counts)}")
        split_students[split] = students
        split_original_ids[split] = ids
        stats[split] = {
            "rows": rows,
            "students": len(students),
            "by_source": dict(sorted(counts.items())),
            "labels": dict(sorted(labels.items())),
            "sha256": digest(output_path),
        }

    overlap = {
        "train_valid_students": len(split_students["train"] & split_students["valid"]),
        "train_test_students": len(split_students["train"] & split_students["test"]),
        "valid_test_students": len(split_students["valid"] & split_students["test"]),
        "train_test_original_examples": len(split_original_ids["train"] & split_original_ids["test"]),
        "valid_test_original_examples": len(split_original_ids["valid"] & split_original_ids["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"Frozen split leakage: {overlap}")
    receipt = {
        "stage": "modern_kt_t1_five_source_split_v1",
        "status": "complete",
        "task": TASK,
        "source_dir": str(input_dir),
        "source_sha256": {split: digest(input_dir / f"{split}.jsonl") for split in EXPECTED},
        "splits": stats,
        "overlap": overlap,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.input_dir, args.output_dir), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
