#!/usr/bin/env python3
"""Mix available C1 datasets into one train/test pair."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def add_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new = dict(row)
        meta = dict(new.get("metadata") or {})
        meta["mix_source"] = source
        new["metadata"] = meta
        out.append(new)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "by_dataset": dict(Counter(str(r.get("dataset", "")) for r in rows)),
        "by_task": dict(Counter(str(r.get("task", "")) for r in rows)),
        "by_source": dict(Counter(str((r.get("metadata") or {}).get("mix_source", "")) for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eedi-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_c1_k12"))
    parser.add_argument("--malrule-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/malrule_c1_clean"))
    parser.add_argument("--eedi-mining-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_mining_c1_ir"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/mixed_c1_no_val"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = []
    train += add_source(read_jsonl(args.eedi_dir / "train.jsonl"), "eedi_response")
    train += add_source(read_jsonl(args.malrule_dir / "train.jsonl"), "malrule")
    train += add_source(read_jsonl(args.eedi_mining_dir / "train.jsonl"), "eedi_mining_ir")

    test = []
    test += add_source(read_jsonl(args.eedi_dir / "test.jsonl"), "eedi_response")
    test += add_source(read_jsonl(args.malrule_dir / "test_cross_template.jsonl"), "malrule_cross")
    test += add_source(read_jsonl(args.eedi_mining_dir / "test.jsonl"), "eedi_mining_ir")

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "test.jsonl", test)
    stats = {"seed": args.seed, "train": summarize(train), "test": summarize(test)}
    (args.out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
