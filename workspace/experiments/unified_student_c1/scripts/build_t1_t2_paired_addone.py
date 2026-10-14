#!/usr/bin/env python3
"""Build the T1 + interaction-matched T2 add-one diagnostic corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TASK_PATTERN = re.compile(r":(predict_correctness|predict_response):")


def event_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    value = str(metadata.get("original_example_id") or row["example_id"])
    return TASK_PATTERN.sub(":TASK:", value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(t1_path: Path, t2_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    t1_keys: set[str] = set()
    t1_rows = 0
    with t1_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            t1_keys.add(event_key(json.loads(line)))
            t1_rows += 1
    if t1_rows != 110_525 or len(t1_keys) != t1_rows:
        raise ValueError(f"T1 support drift: rows={t1_rows} keys={len(t1_keys)}")

    paired_t2_rows = 0
    sources: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        with t1_path.open() as source_handle:
            for line in source_handle:
                if line.strip():
                    handle.write(line if line.endswith("\n") else line + "\n")
        with t2_path.open() as source_handle:
            for line in source_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if event_key(row) in t1_keys:
                    handle.write(line if line.endswith("\n") else line + "\n")
                    paired_t2_rows += 1
                    sources[str((row.get("metadata") or {}).get("mix_source"))] += 1
    if paired_t2_rows != 52_240:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Paired T2 support drift: {paired_t2_rows}")
    if sources != Counter({"dbe_kt22": 3_690, "eedi_response": 16_720, "foundationalassist": 31_830}):
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Paired T2 source drift: {sources}")

    receipt = {
        "stage": "t1_t2_paired_addone_data_v1",
        "status": "complete",
        "rows": t1_rows + paired_t2_rows,
        "t1_rows": t1_rows,
        "paired_t2_rows": paired_t2_rows,
        "paired_t2_sources": dict(sorted(sources.items())),
        "t1_exposure_matches_taskonly": True,
        "total_update_matched": False,
        "output_sha256": sha256(output_path),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--t2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.t1, args.t2, args.output, args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
