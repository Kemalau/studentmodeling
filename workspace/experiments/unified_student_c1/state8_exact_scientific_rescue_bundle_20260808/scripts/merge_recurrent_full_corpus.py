#!/usr/bin/env python3
"""Merge audited recurrent train/valid splits for final post-selection fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def merge(train: Path, valid: Path, output: Path, manifest: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as target:
        for source_split, path in (("train", train), ("valid", valid)):
            for row in rows(path):
                example_id = str(row.get("example_id") or "")
                if not example_id or example_id in seen:
                    raise ValueError(f"Missing/duplicate example_id: {example_id}")
                seen.add(example_id)
                if "<student_history>" in str(row.get("decode_text") or "").lower():
                    raise ValueError(f"Decoder history leak in {example_id}")
                if str(row.get("decode_text") or "").count("<STUDENT_STATE>") != 1:
                    raise ValueError(f"Invalid decoder carrier count in {example_id}")
                item = dict(row)
                item["final_fit_source_split"] = source_split
                target.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                counts["rows"] += 1
                counts[f"source_split/{source_split}"] += 1
                counts[f"task/{row['task']}"] += 1
                counts[f"source/{row['source']}"] += 1
    payload = {
        "version": 1,
        "purpose": "post_selection_final_fit_train_plus_validation",
        "inputs": {
            "train": {"path": str(train), "sha256": sha256(train)},
            "valid": {"path": str(valid), "sha256": sha256(valid)},
        },
        "output": {"path": str(output), "sha256": sha256(output)},
        "unique_examples": len(seen),
        "counts": dict(sorted(counts.items())),
        "selection_data_note": (
            "Loss/projection selection is completed before this merged corpus is used; "
            "the formal test remains untouched."
        ),
    }
    if payload["unique_examples"] != counts["rows"]:
        raise AssertionError("Merged row count differs from unique example count")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(args.train_jsonl, args.valid_jsonl, args.output_jsonl, args.manifest),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
