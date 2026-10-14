#!/usr/bin/env python3
"""Keep the first N rows per task from a shuffled evaluation JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, required=True)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row["task"])
            if counts[task] >= args.per_task:
                continue
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[task] += 1
    print(json.dumps(dict(counts), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
