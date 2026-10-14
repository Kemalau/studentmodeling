#!/usr/bin/env python3
"""Build deterministic recurrent-update examples from method-small rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from recurrent_student_model import (
    assert_decode_text_is_closed,
    build_observation_matched_direct_text,
    build_update_text,
    extract_direct_decode_text,
    truncate_direct_history,
)


SUPPORTED_TASKS = {
    "malrule_next_response",
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "predict_prospective_error",
    "predict_error_persistence",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def source(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")


def stable_score(seed: int, row: dict[str, Any]) -> int:
    payload = f"{seed}\x1f{row.get('example_id')}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def make_record(row: dict[str, Any], unroll: int) -> dict[str, Any] | None:
    history = list(row.get("history") or [])
    if len(history) < unroll:
        return None
    task = str(row.get("task"))
    if task not in SUPPORTED_TASKS:
        return None
    decode_text = extract_direct_decode_text(str(row["prompt"]))
    assert_decode_text_is_closed(decode_text)
    updates = history[-unroll:]
    update_texts = [build_update_text(interaction) for interaction in updates]
    direct_text = truncate_direct_history(str(row["prompt"]), unroll)
    if any("<student_history>" in text.lower() for text in update_texts):
        raise AssertionError("Update text unexpectedly contains a raw-history container")
    return {
        "example_id": str(row["example_id"]),
        "student_id": str(row.get("student_id")),
        "task": task,
        "source": source(row),
        "unroll": unroll,
        "update_texts": update_texts,
        "decode_text": decode_text,
        "direct_text": direct_text,
        "target": str(row["target"]),
        "target_label": row.get("target_label"),
        "candidate_labels": list(row.get("candidate_labels") or []),
        "candidate_targets": dict(row.get("candidate_targets") or {}),
        "history_length": len(history),
    }


def build_split(path: Path, limit: int, unroll: int, seed: int) -> tuple[list[dict], dict]:
    candidates = []
    skipped = Counter()
    for row in read_jsonl(path):
        record = make_record(row, unroll)
        if record is None:
            skipped["short_or_unsupported"] += 1
            continue
        candidates.append((stable_score(seed, row), record))
    candidates.sort(key=lambda item: (item[0], item[1]["example_id"]))
    selected = [item[1] for item in candidates[:limit]]
    if len(selected) < limit:
        raise ValueError(f"Only {len(selected)} eligible rows in {path}; requested {limit}")
    counts = Counter((record["task"], record["source"]) for record in selected)
    stats = {
        "input": str(path),
        "input_sha256": sha256(path),
        "eligible": len(candidates),
        "selected": len(selected),
        "unroll": unroll,
        "unique_students": len({record["student_id"] for record in selected}),
        "counts": {f"{task}/{src}": value for (task, src), value in sorted(counts.items())},
        "skipped": dict(skipped),
    }
    return selected, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--valid-limit", type=int, default=300)
    parser.add_argument("--unroll", type=int, choices=(1, 4), default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    train, train_stats = build_split(args.train_jsonl, args.train_limit, args.unroll, args.seed)
    valid, valid_stats = build_split(args.valid_jsonl, args.valid_limit, args.unroll, args.seed + 1)
    train_students = {row["student_id"] for row in train}
    valid_students = {row["student_id"] for row in valid}
    overlap = train_students & valid_students
    if overlap:
        raise ValueError(f"Train/valid recurrent student overlap: {sorted(overlap)[:5]}")

    train_path = args.output_dir / "train.jsonl"
    valid_path = args.output_dir / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    stats = {
        "version": 1,
        "train": train_stats,
        "valid": valid_stats,
        "train_valid_student_overlap": 0,
        "outputs": {
            "train_sha256": sha256(train_path),
            "valid_sha256": sha256(valid_path),
        },
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
