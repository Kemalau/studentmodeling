#!/usr/bin/env python3
"""Select a deterministic task/source-stratified recurrent manifest subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def score(seed: int, example_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}\x1f{example_id}".encode(), digest_size=8).digest(),
        "big",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    groups = defaultdict(list)
    with args.input_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                groups[(str(row["task"]), str(row["source"]))].append(row)
    total = sum(len(rows) for rows in groups.values())
    if args.limit > total:
        raise ValueError(f"Requested {args.limit} from only {total} rows")
    allocations = {}
    remaining = args.limit
    keys = sorted(groups)
    for index, key in enumerate(keys):
        if index == len(keys) - 1:
            take = remaining
        else:
            take = round(args.limit * len(groups[key]) / total)
            take = max(1, min(take, len(groups[key]), remaining - (len(keys) - index - 1)))
        allocations[key] = take
        remaining -= take
    selected = []
    for key in keys:
        ranked = sorted(
            groups[key], key=lambda row: (score(args.seed, str(row["example_id"])), str(row["example_id"]))
        )
        selected.extend(ranked[: allocations[key]])
    selected.sort(key=lambda row: (score(args.seed + 1, str(row["example_id"])), str(row["example_id"])))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected": len(selected),
                "allocations": {f"{task}/{source}": value for (task, source), value in allocations.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
