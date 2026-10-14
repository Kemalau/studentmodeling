#!/usr/bin/env python3
"""Fail-closed remote progress helpers for state-only T3 generation shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def row_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": str(row["example_id"]),
        "task": str(row["task"]),
        "source": str(row["source"]),
        "protocol": str(row["protocol"]),
        "delta": row.get("delta"),
        "target": row["target"],
    }


def progress_manifest(
    rows: list[dict[str, Any]], progress_key: str
) -> dict[str, Any]:
    signatures = [row_signature(row) for row in rows]
    encoded = json.dumps(
        signatures, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "stage": "state8_t3_generation_progress_v1",
        "progress_key": progress_key,
        "rows": len(rows),
        "row_signatures_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def load_progress(
    progress_dir: Path,
    rows: list[dict[str, Any]],
    progress_key: str,
) -> dict[str, dict[str, Any]]:
    if not progress_key:
        raise ValueError("Progress key must be non-empty")
    expected_manifest = progress_manifest(rows, progress_key)
    manifest_path = progress_dir / "manifest.json"
    records_path = progress_dir / "records.jsonl"
    progress_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected_manifest:
            raise ValueError("Progress manifest drift")
    else:
        if records_path.exists():
            raise ValueError("Progress records exist without a manifest")
        atomic_write_json(manifest_path, expected_manifest)
    if not records_path.exists():
        return {}
    expected = {str(row["example_id"]): row_signature(row) for row in rows}
    records: dict[str, dict[str, Any]] = {}
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = str(record.get("example_id"))
            if example_id not in expected:
                raise ValueError(f"Unknown progress example {example_id}")
            if example_id in records:
                raise ValueError(f"Duplicate progress example {example_id}")
            signature = {key: record.get(key) for key in expected[example_id]}
            if signature != expected[example_id] or "raw_output" not in record:
                raise ValueError(f"Progress row drift at line {line_number}")
            records[example_id] = record
    return records
