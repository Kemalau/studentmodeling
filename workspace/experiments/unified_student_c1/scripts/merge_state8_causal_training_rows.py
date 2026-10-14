#!/usr/bin/env python3
"""Merge disjoint causal training datasets with a hash-bound receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    merged = []
    sources = []
    for path in args.input:
        values = list(rows(path))
        merged.extend(values)
        sources.append({"path": str(path), "sha256": sha256(path), "rows": len(values)})
    ids = [str(row["pair_id"]) for row in merged]
    if not merged or len(ids) != len(set(ids)):
        raise ValueError("Merged causal rows are empty or have duplicate pair IDs")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output_jsonl)
    receipt = {
        "stage": "state8_causal_training_merge_v1", "status": "passed",
        "sources": sources, "rows": len(merged),
        "target_kind_counts": dict(sorted(Counter(str(row.get("causal_target_kind")) for row in merged).items())),
        "source_counts": dict(sorted(Counter(str(row.get("source") or "malrule") for row in merged).items())),
        "output_jsonl": str(args.output_jsonl), "output_sha256": sha256(args.output_jsonl),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
