#!/usr/bin/env python3
"""Materialize an audited recurrent subset in its preregistered order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("source_train_sha256") != sha256(args.source_jsonl):
        raise ValueError("Index manifest source hash drift")
    if manifest.get("formal_eval_sha256") != sha256(args.formal_eval_jsonl):
        raise ValueError("Index manifest formal-eval hash drift")
    rows = read(args.source_jsonl)
    indices = [int(index) for index in manifest.get("indices") or []]
    if not indices or len(indices) != int(manifest.get("segment_rows", -1)):
        raise ValueError("Index manifest cardinality drift")
    if len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= len(rows):
        raise ValueError("Index manifest indices are invalid")
    selected = [rows[index] for index in indices]
    selected_ids = [str(row["example_id"]) for row in selected]
    observed_id_hash = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    if observed_id_hash != manifest.get("selected_example_ids_sha256"):
        raise ValueError("Index manifest example order drift")
    formal_ids = {str(row["example_id"]) for row in read(args.formal_eval_jsonl)}
    overlap = formal_ids.intersection(selected_ids)
    if overlap:
        raise ValueError(f"Materialized train/test overlap: {sorted(overlap)[:5]}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = {
        "stage": "materialized_recurrent_index_manifest_v1",
        "rows": len(selected),
        "source_sha256": sha256(args.source_jsonl),
        "manifest_sha256": sha256(args.manifest),
        "output_sha256": sha256(args.output_jsonl),
        "formal_eval_sha256": sha256(args.formal_eval_jsonl),
        "formal_example_id_overlap": 0,
        "selected_example_ids_sha256": observed_id_hash,
    }
    receipt_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
