#!/usr/bin/env python3
"""Fail-closed audit for unified recurrent student-state manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STATE_TOKEN = "<STUDENT_STATE>"
REQUIRED_TASKS = {
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "predict_prospective_error",
    "predict_error_persistence",
}
NON_STUDENT_SENTINELS = {
    "eedi_mining::NO_STUDENT_HISTORY",
    "None",
    "null",
    "",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def audit_split(path: Path) -> tuple[dict[str, Any], set[str], dict[str, set[str]]]:
    tasks: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    semantics: Counter[str] = Counter()
    examples: set[str] = set()
    students: dict[str, set[str]] = defaultdict(set)
    rows = 0
    cold_start_rows = 0

    for line_number, row in read_jsonl(path):
        rows += 1
        required = {
            "example_id",
            "student_id",
            "task",
            "source",
            "sequence_semantics",
            "update_texts",
            "decode_text",
            "direct_text",
            "target",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Missing fields at {path}:{line_number}: {missing}")

        example_id = str(row["example_id"])
        if example_id in examples:
            raise ValueError(f"Duplicate example_id at {path}:{line_number}: {example_id}")
        examples.add(example_id)

        task = str(row["task"])
        source = str(row["source"])
        semantic = str(row["sequence_semantics"])
        student_id = str(row["student_id"])
        tasks[task] += 1
        sources[source] += 1
        semantics[semantic] += 1
        students[source].add(student_id)

        decode = str(row["decode_text"])
        direct = str(row["direct_text"])
        full_context = str(row.get("full_context_text") or direct)
        updates = list(row["update_texts"])
        if "<student_history>" in decode.lower():
            raise ValueError(f"Raw-history leak in decoder at {path}:{line_number}")
        if decode.count(STATE_TOKEN) != 1:
            raise ValueError(f"Decoder carrier count != 1 at {path}:{line_number}")
        if direct.count(STATE_TOKEN) != 1:
            raise ValueError(f"Direct carrier count != 1 at {path}:{line_number}")
        if full_context.count(STATE_TOKEN) != 1:
            raise ValueError(f"Full-context carrier count != 1 at {path}:{line_number}")
        for update_index, update in enumerate(updates):
            update = str(update)
            if "<student_history>" in update.lower():
                raise ValueError(
                    f"Raw-history container in update {update_index} at {path}:{line_number}"
                )
            if update.count(STATE_TOKEN) != 2:
                raise ValueError(
                    f"Update carrier count != 2 in update {update_index} at "
                    f"{path}:{line_number}"
                )
        if semantic == "cold_start_no_observation":
            cold_start_rows += 1
            if updates:
                raise ValueError(f"Cold-start row has updates at {path}:{line_number}")
        elif not updates:
            raise ValueError(f"Non-cold-start row lacks updates at {path}:{line_number}")

    missing_tasks = sorted(REQUIRED_TASKS - tasks.keys())
    if missing_tasks:
        raise ValueError(f"Required tasks absent from {path}: {missing_tasks}")
    summary = {
        "path": str(path),
        "sha256": sha256(path),
        "rows": rows,
        "unique_examples": len(examples),
        "tasks": dict(sorted(tasks.items())),
        "sources": dict(sorted(sources.items())),
        "sequence_semantics": dict(sorted(semantics.items())),
        "cold_start_rows": cold_start_rows,
        "decoder_history_leaks": 0,
        "carrier_errors": 0,
    }
    return summary, examples, students


def audit_manifests(train_path: Path, valid_path: Path) -> dict[str, Any]:
    train, train_examples, train_students = audit_split(train_path)
    valid, valid_examples, valid_students = audit_split(valid_path)
    example_overlap = train_examples & valid_examples
    if example_overlap:
        raise ValueError(f"Train/valid example overlap: {sorted(example_overlap)[:5]}")

    student_overlap: dict[str, int] = {}
    ignored_sentinels: dict[str, list[str]] = {}
    for source in sorted(set(train_students) | set(valid_students)):
        overlap = train_students[source] & valid_students[source]
        ignored = overlap & NON_STUDENT_SENTINELS
        real_overlap = overlap - NON_STUDENT_SENTINELS
        if real_overlap:
            raise ValueError(
                f"Train/valid real-student overlap for {source}: {sorted(real_overlap)[:5]}"
            )
        student_overlap[source] = 0
        if ignored:
            ignored_sentinels[source] = sorted(ignored)

    return {
        "version": 1,
        "status": "passed",
        "required_tasks": sorted(REQUIRED_TASKS),
        "train": train,
        "valid": valid,
        "train_valid_example_overlap": 0,
        "train_valid_real_student_overlap": student_overlap,
        "ignored_non_student_sentinels": ignored_sentinels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = audit_manifests(args.train_jsonl, args.valid_jsonl)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
