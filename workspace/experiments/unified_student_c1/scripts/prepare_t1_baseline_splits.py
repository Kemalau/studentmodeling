#!/usr/bin/env python3
"""Create deduplicated student-disjoint T1 train/valid files for KT baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--valid-percent", type=int, default=10)
    args = parser.parse_args()

    unique = {}
    with args.train_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "predict_correctness":
                continue
            original_id = str((row.get("metadata") or {}).get("original_example_id") or row["example_id"])
            unique.setdefault(original_id, row)
    train, valid = [], []
    for row in unique.values():
        bucket = int(hashlib.sha256(str(row["student_id"]).encode("utf-8")).hexdigest()[:8], 16) % 100
        (valid if bucket < args.valid_percent else train).append(row)
    test = []
    with args.test_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("task") == "predict_correctness":
                    test.append(row)
    write(args.out_dir / "train.jsonl", train)
    write(args.out_dir / "valid.jsonl", valid)
    write(args.out_dir / "test.jsonl", test)
    print(json.dumps({"train": len(train), "valid": len(valid), "test": len(test)}, indent=2))


if __name__ == "__main__":
    main()
