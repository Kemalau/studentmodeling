#!/usr/bin/env python3
"""Prepare Eedi-Mining IR mirror rows for C1 misconception diagnosis.

The public HF mirror is already expanded into query -> misconception qrels.
For the quick mixed C1 run we keep evaluation tractable by giving each row a
deterministic closed candidate set: gold + N negative misconception ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


STATE_TOKEN = "<STUDENT_STATE>"


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


def make_prompt(query: str) -> str:
    return (
        "You are a student misconception diagnosis system for mathematics.\n"
        "<task>diagnose_misconception</task>\n"
        "<instruction>Given a math question, the correct answer, and a student's incorrect answer, "
        "diagnose the misconception that most likely caused the error. Return exactly one valid XML-style tag.</instruction>\n"
        "<student_history>\n</student_history>\n"
        f"{STATE_TOKEN}\n"
        "<query>\n"
        f"{query}\n"
        "</query>\n"
        "Answer:"
    )


def build_rows(
    raw_dir: Path,
    split_folds: set[int],
    split_name: str,
    num_negatives: int,
    seed: int,
) -> list[dict[str, Any]]:
    corpus = read_jsonl(raw_dir / "corpus.jsonl")
    id_to_text = {str(row["id_"]): str(row["text"]) for row in corpus}
    all_ids = sorted(id_to_text)
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(raw_dir / "queries_w_rels.jsonl"):
        fold = int(raw["fold"])
        if fold not in split_folds:
            continue
        gold = str(raw["mid"])
        pool = [mid for mid in all_ids if mid != gold]
        qid_hash = int(hashlib.sha256(str(raw["qid"]).encode("utf-8")).hexdigest()[:12], 16)
        row_rng = random.Random(seed + fold * 1_000_003 + qid_hash)
        negs = row_rng.sample(pool, k=min(num_negatives, len(pool)))
        labels = [gold] + negs
        rng.shuffle(labels)
        candidate_targets = {mid: f"<misconception>{id_to_text[mid]}</misconception>" for mid in labels}
        query = str(raw["query"])
        rows.append(
            {
                "example_id": f"eedi_mining_ir:{split_name}:{raw['qid']}",
                "dataset": "eedi_mining_ir",
                "split": split_name,
                "task": "diagnose_misconception",
                "student_id": "NO_STUDENT_HISTORY",
                "history": [],
                "query": {
                    "qid": str(raw["qid"]),
                    "query_text": query,
                    "gold_misconception_id": gold,
                    "gold_misconception_name": id_to_text[gold],
                },
                "target_label": gold,
                "target": f"<misconception>{id_to_text[gold]}</misconception>",
                "candidate_labels": labels,
                "candidate_targets": candidate_targets,
                "prompt": make_prompt(query),
                "metadata": {
                    "source": "cdtmc/eedi-ir",
                    "fold": fold,
                    "question_id": str(raw["qid"]),
                    "skill_id": "",
                    "misconception_id": gold,
                    "candidate_set": f"gold_plus_{len(labels)-1}_negatives",
                },
            }
        )
    rows.sort(key=lambda r: r["example_id"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/unified_student_c1/data/raw/eedi_mining_2024_hf_ir"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_mining_c1_ir"))
    parser.add_argument("--test-fold", type=int, default=4)
    parser.add_argument("--num-negatives", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_folds = {0, 1, 2, 3, 4} - {args.test_fold}
    test_folds = {args.test_fold}
    train = build_rows(args.raw_dir, train_folds, "train", args.num_negatives, args.seed)
    test = build_rows(args.raw_dir, test_folds, "test", args.num_negatives, args.seed)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "test.jsonl", test)
    stats = {
        "source": str(args.raw_dir),
        "test_fold": args.test_fold,
        "train_folds": sorted(train_folds),
        "num_negatives": args.num_negatives,
        "train_rows": len(train),
        "test_rows": len(test),
    }
    (args.out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
