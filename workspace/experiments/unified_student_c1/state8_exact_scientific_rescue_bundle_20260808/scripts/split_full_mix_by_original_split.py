#!/usr/bin/env python3
"""Restore the frozen train/validation split from the prompt-complete full mix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_full_mix(input_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: output_dir / f"{split}.jsonl" for split in ("train", "valid")}
    counts: Counter[str] = Counter()
    examples: set[str] = set()
    students: dict[tuple[str, str], set[str]] = defaultdict(set)
    handles = {split: path.open("w", encoding="utf-8") for split, path in outputs.items()}
    try:
        with input_path.open(encoding="utf-8") as source_handle:
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("metadata") or {}
                split = str(metadata.get("original_split") or "")
                if split not in handles:
                    raise ValueError(
                        f"Unsupported original_split={split!r} at {input_path}:{line_number}"
                    )
                if not str(row.get("prompt") or "").strip():
                    raise ValueError(f"Missing prompt at {input_path}:{line_number}")
                example_id = str(row.get("example_id") or "")
                if not example_id or example_id in examples:
                    raise ValueError(f"Missing/duplicate example_id at line {line_number}: {example_id}")
                examples.add(example_id)
                source = str(metadata.get("mix_source") or row.get("dataset") or "unknown")
                student = str(row.get("student_id") or "")
                students[(split, source)].add(student)
                row["split"] = split
                handles[split].write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                counts[f"{split}/rows"] += 1
                counts[f"{split}/task/{row['task']}"] += 1
                counts[f"{split}/source/{source}"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    overlap = {}
    sources = sorted({source for _, source in students})
    for source in sources:
        shared = students[("train", source)] & students[("valid", source)]
        overlap[source] = len(shared)
        if shared:
            raise ValueError(
                f"Frozen split has train/valid student overlap for {source}: "
                f"{sorted(shared)[:5]}"
            )
    total = counts["train/rows"] + counts["valid/rows"]
    if total != len(examples):
        raise AssertionError("Output row count does not match unique input examples")
    result = {
        "version": 1,
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "outputs": {
            split: {
                "path": str(path),
                "sha256": sha256(path),
                "rows": counts[f"{split}/rows"],
            }
            for split, path in outputs.items()
        },
        "total_rows": total,
        "counts": dict(sorted(counts.items())),
        "train_valid_student_overlap": overlap,
    }
    (output_dir / "split_stats.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(split_full_mix(args.input_jsonl, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
