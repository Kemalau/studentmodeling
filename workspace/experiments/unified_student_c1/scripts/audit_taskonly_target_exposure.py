#!/usr/bin/env python3
"""Audit exact target-task exposure parity for task-only Qwen controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = (
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "predict_prospective_error",
    "predict_error_persistence",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def target_signature(row: dict[str, Any]) -> str:
    payload = {
        "task": row.get("task"),
        "target": row.get("target"),
        "target_label": row.get("target_label"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def digest_values(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def audit(
    unified_jsonl: Path,
    unified_manifest_path: Path,
    taskonly_root: Path,
    taskonly_manifest_path: Path,
) -> dict[str, Any]:
    unified_manifest = read_json(unified_manifest_path)
    taskonly_manifest = read_json(taskonly_manifest_path)
    expected_unified = {
        key.removeprefix("task/"): int(value)
        for key, value in (unified_manifest.get("counts") or {}).items()
        if key.startswith("task/")
    }
    if set(expected_unified) != set(TASKS):
        raise ValueError(f"Unified task-count coverage drift: {expected_unified}")

    taskonly_rows: dict[str, dict[str, str]] = {}
    taskonly_counts: Counter[str] = Counter()
    for task in TASKS:
        path = taskonly_root / f"{task}.jsonl"
        records: dict[str, str] = {}
        for row in read_jsonl(path):
            if row.get("task") != task:
                raise ValueError(f"Task drift in {path}: {row.get('task')!r}")
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in records:
                raise ValueError(f"Missing or duplicate task-only example ID: {task}: {example_id!r}")
            records[example_id] = target_signature(row)
        manifest_task = (taskonly_manifest.get("tasks") or {}).get(task) or {}
        if len(records) != int(manifest_task.get("train_rows", -1)):
            raise ValueError(f"Task-only manifest count drift for {task}: {len(records)}")
        taskonly_rows[task] = records
        taskonly_counts[task] = len(records)

    unified_ids: dict[str, set[str]] = {task: set() for task in TASKS}
    unified_counts: Counter[str] = Counter()
    mismatches: Counter[str] = Counter()
    unmatched_unified: Counter[str] = Counter()
    for row in read_jsonl(unified_jsonl):
        task = str(row.get("task") or "")
        if task not in taskonly_rows:
            raise ValueError(f"Unexpected unified task: {task!r}")
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in unified_ids[task]:
            raise ValueError(f"Missing or duplicate unified example ID: {task}: {example_id!r}")
        unified_ids[task].add(example_id)
        unified_counts[task] += 1
        expected_signature = taskonly_rows[task].get(example_id)
        if expected_signature is None:
            unmatched_unified[task] += 1
        elif expected_signature != target_signature(row):
            mismatches[task] += 1

    rows: dict[str, Any] = {}
    for task in TASKS:
        taskonly_ids = set(taskonly_rows[task])
        only_unified = unified_ids[task] - taskonly_ids
        only_taskonly = taskonly_ids - unified_ids[task]
        if unified_counts[task] != expected_unified[task]:
            raise ValueError(f"Unified manifest count drift for {task}: {unified_counts[task]}")
        rows[task] = {
            "unified_rows": unified_counts[task],
            "taskonly_rows": taskonly_counts[task],
            "unified_unique_ids": len(unified_ids[task]),
            "taskonly_unique_ids": len(taskonly_ids),
            "unified_id_sha256": digest_values(unified_ids[task]),
            "taskonly_id_sha256": digest_values(taskonly_ids),
            "unified_only": len(only_unified),
            "taskonly_only": len(only_taskonly),
            "target_signature_mismatches": mismatches[task],
            "id_set_equal": not only_unified and not only_taskonly,
            "target_supervision_equal": mismatches[task] == 0,
        }
    passed = all(
        row["id_set_equal"]
        and row["target_supervision_equal"]
        and row["unified_rows"] == row["taskonly_rows"]
        for row in rows.values()
    )
    if not passed:
        raise ValueError(f"Target-task exposure parity failed: {rows}")
    return {
        "stage": "taskonly_qwen_target_exposure_audit_v1",
        "status": "complete",
        "matching_unit": "unique example_id plus task/target/target_label signature",
        "target_task_exposure_exact_match": True,
        "total_unified_rows": sum(unified_counts.values()),
        "unified_manifest": str(unified_manifest_path),
        "taskonly_manifest": str(taskonly_manifest_path),
        "tasks": rows,
        "interpretation": (
            "Each one-epoch task-only arm sees exactly the same unique target-task "
            "examples and target supervision once as the unified final-fit epoch. "
            "This is target-task-exposure matched, not total-update matched; matching "
            "total updates would repeat target examples and is a separate repetition-dose control."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-jsonl", type=Path, required=True)
    parser.add_argument("--unified-manifest", type=Path, required=True)
    parser.add_argument("--taskonly-root", type=Path, required=True)
    parser.add_argument("--taskonly-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        args.unified_jsonl,
        args.unified_manifest,
        args.taskonly_root,
        args.taskonly_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
