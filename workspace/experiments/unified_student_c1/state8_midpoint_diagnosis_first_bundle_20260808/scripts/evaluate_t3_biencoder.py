#!/usr/bin/env python3
"""Evaluate an off-the-shelf BGE bi-encoder on Eedi-Mining T3 candidates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "")
            if row.get("task") == "diagnose_misconception" and source == "eedi_mining":
                rows.append(row)
    return rows


def clean_candidate(value: str) -> str:
    return re.sub(r"</?misconception>", "", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    rows = read_rows(args.test_jsonl)
    candidates = sorted(
        {
            clean_candidate(str(target))
            for row in rows
            for target in (row.get("candidate_targets") or {}).values()
        }
    )
    candidate_index = {text: index for index, text in enumerate(candidates)}
    model = SentenceTransformer(args.model, device="cpu")
    query_texts = [
        "Represent this sentence for searching relevant passages: " + str((row.get("query") or {}).get("query_text") or "")
        for row in rows
    ]
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    candidate_embeddings = model.encode(
        candidates,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    correct = 0
    reciprocal_ranks = []
    recall_at = {1: 0, 5: 0, 10: 0}
    for row_index, row in enumerate(rows):
        labels = [str(label) for label in row["candidate_labels"]]
        texts = [clean_candidate(str(row["candidate_targets"][label])) for label in labels]
        indices = [candidate_index[text] for text in texts]
        scores = candidate_embeddings[indices] @ query_embeddings[row_index]
        order = np.argsort(-scores)
        gold_position = labels.index(str(row["target_label"]))
        rank = int(np.where(order == gold_position)[0][0]) + 1
        correct += int(rank == 1)
        reciprocal_ranks.append(1.0 / rank)
        for cutoff in recall_at:
            recall_at[cutoff] += int(rank <= cutoff)
    result = {
        "model": args.model,
        "task": "diagnose_misconception/eedi_mining",
        "n": len(rows),
        "accuracy": correct / max(1, len(rows)),
        "mrr": sum(reciprocal_ranks) / max(1, len(rows)),
        **{f"recall_at_{cutoff}": value / max(1, len(rows)) for cutoff, value in recall_at.items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
