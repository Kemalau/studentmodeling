#!/usr/bin/env python3
"""Build unified recurrent-state records without inventing chronology.

Longitudinal sources use their observed interaction order.  Eedi Response is a
non-temporal profile and retains the source's deterministic profile order only
as a serialization convention.  Eedi-Mining contributes one observed wrong
response followed by a state-only misconception query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from recurrent_student_model import (
    STATE_TOKEN,
    build_update_text,
    extract_direct_decode_text,
    truncate_direct_history,
)


EEDI_MINING_SOURCE = "eedi_mining"
EEDI_RESPONSE_SOURCE = "eedi_response"


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


def common_record(
    row: dict[str, Any],
    update_texts: list[str],
    direct_text: str,
    full_context_text: str,
    decode_text: str,
    semantics: str,
    observed_interactions: int,
) -> dict[str, Any]:
    if any(text.count(STATE_TOKEN) != 2 for text in update_texts):
        raise AssertionError("Each update must contain exactly two state carriers")
    if (
        direct_text.count(STATE_TOKEN) != 1
        or full_context_text.count(STATE_TOKEN) != 1
        or decode_text.count(STATE_TOKEN) != 1
    ):
        raise AssertionError(
            "Direct, full-context, and decode views must each contain one state carrier"
        )
    if "<student_history>" in decode_text.lower():
        raise AssertionError("State-only decoder contains raw history")
    return {
        "example_id": str(row["example_id"]),
        "student_id": str(row.get("student_id")),
        "task": str(row["task"]),
        "source": source(row),
        "sequence_semantics": semantics,
        "observed_interactions": observed_interactions,
        "update_texts": update_texts,
        "decode_text": decode_text,
        "direct_text": direct_text,
        "full_context_text": full_context_text,
        "target": str(row["target"]),
        "target_label": row.get("target_label"),
        "candidate_labels": list(row.get("candidate_labels") or []),
        "candidate_targets": dict(row.get("candidate_targets") or {}),
        "history_length": len(row.get("history") or []),
        "original_metadata": row.get("metadata") or {},
    }


def make_history_record(row: dict[str, Any], max_updates: int) -> dict[str, Any] | None:
    history = list(row.get("history") or [])
    if not history:
        return common_record(
            row,
            [],
            str(row["prompt"]),
            str(row["prompt"]),
            extract_direct_decode_text(str(row["prompt"])),
            "cold_start_no_observation",
            0,
        )
    src = source(row)
    if src == EEDI_RESPONSE_SOURCE:
        # This source is a deterministic non-temporal profile, not a sequence
        # claim.  It still obeys the same fixed update budget as every source.
        interactions = history[-max_updates:]
        semantics = "non_temporal_profile_deterministic_serialization"
    else:
        interactions = history[-max_updates:]
        semantics = "chronological_trajectory"
    update_texts = [build_update_text(interaction) for interaction in interactions]
    direct_text = truncate_direct_history(str(row["prompt"]), len(interactions))
    decode_text = extract_direct_decode_text(str(row["prompt"]))
    return common_record(
        row,
        update_texts,
        direct_text,
        str(row["prompt"]),
        decode_text,
        semantics,
        len(interactions),
    )


def make_eedi_mining_record(row: dict[str, Any]) -> dict[str, Any]:
    query_text = str((row.get("query") or {}).get("query_text") or "").strip()
    if not query_text:
        raise ValueError(f"Eedi-Mining row lacks query_text: {row.get('example_id')}")
    prompt = str(row["prompt"])
    history_open = "<student_history>"
    history_close = "</student_history>"
    start = prompt.find(history_open)
    end = prompt.find(history_close)
    if start < 0 or end < 0:
        raise ValueError("Eedi-Mining prompt lacks history container")
    prefix = prompt[:start]
    generic_query = (
        "<query>\nDiagnose the misconception expressed in the observed "
        "mathematics interaction stored in the student state.\n</query>\nAnswer:"
    )
    observation = {
        "interaction_id": str(row["example_id"]),
        "question_id": str((row.get("metadata") or {}).get("question_id") or ""),
        "question_text": query_text,
    }
    update_texts = [build_update_text(observation)]
    direct_text = (
        f"{prefix}{history_open}\nTrial 1:\nObserved mathematics interaction:\n"
        f"{query_text}\n{history_close}\n{STATE_TOKEN}\n{generic_query}"
    )
    decode_text = f"{prefix}{STATE_TOKEN}\n{generic_query}"
    return common_record(
        row,
        update_texts,
        direct_text,
        direct_text,
        decode_text,
        "single_observed_wrong_response",
        1,
    )


def build_split(path: Path, max_updates: int, seed: int, limit: int | None) -> tuple[list[dict], dict]:
    records = []
    skipped = Counter()
    for row in read_jsonl(path):
        if source(row) == EEDI_MINING_SOURCE:
            record = make_eedi_mining_record(row)
        else:
            record = make_history_record(row, max_updates)
        records.append((stable_score(seed, row), record))
    records.sort(key=lambda item: (item[0], item[1]["example_id"]))
    if limit is not None:
        records = records[:limit]
    selected = [record for _, record in records]
    counts = Counter(
        (record["task"], record["source"], record["sequence_semantics"])
        for record in selected
    )
    stats = {
        "input": str(path),
        "input_sha256": sha256(path),
        "selected": len(selected),
        "max_updates": max_updates,
        "counts": {
            "/".join(key): value for key, value in sorted(counts.items())
        },
        "skipped": dict(sorted(skipped.items())),
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
    parser.add_argument("--max-updates", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    train, train_stats = build_split(
        args.train_jsonl, args.max_updates, args.seed, args.train_limit
    )
    valid, valid_stats = build_split(
        args.valid_jsonl, args.max_updates, args.seed + 1, args.valid_limit
    )
    train_path = args.output_dir / "train.jsonl"
    valid_path = args.output_dir / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    stats = {
        "version": 3,
        "train": train_stats,
        "valid": valid_stats,
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
