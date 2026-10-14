#!/usr/bin/env python3
"""Build deterministic donor/recipient pairs for paired state-swap training.

The donor supplies a student history and misconception.  The recipient supplies
the current question.  A pair is retained only when the donor misconception
has a distinct, legal, non-correct answer on the recipient question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK = "malrule_next_response"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") == TASK:
                rows.append(row)
    if not rows:
        raise ValueError(f"No {TASK} rows in {path}")
    return rows


def label(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("malrule_id") or "NONE")


def family(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("family") or "NONE")


def donor_target(donor: dict[str, Any], recipient: dict[str, Any]) -> str | None:
    value = (recipient.get("metadata") or {}).get("query_answers_by_malrule", {}).get(label(donor))
    return None if value is None else str(value)


def valid_pair(donor: dict[str, Any], recipient: dict[str, Any]) -> tuple[bool, str | None]:
    if donor["example_id"] == recipient["example_id"]:
        return False, None
    if label(donor) == "NONE" or label(recipient) == "NONE":
        return False, None
    if label(donor) == label(recipient) or family(donor) != family(recipient):
        return False, None
    target_a = donor_target(donor, recipient)
    target_b = str(recipient.get("target_label"))
    correct = str((recipient.get("query") or {}).get("correct_answer"))
    candidates = {str(value) for value in recipient.get("candidate_labels", [])}
    valid = (
        target_a is not None
        and target_a in candidates
        and target_b in candidates
        and target_a != target_b
        and target_a != correct
        and target_b != correct
    )
    return valid, target_a


def query_key(row: dict[str, Any]) -> str:
    """Return a stable key for the mathematical query, not the student row.

    Multiple generated student rows can share a question.  A multi-query
    curriculum must expose one donor state to genuinely different questions;
    counting a second row for the same question would leave the one-state--one-
    query shortcut intact.
    """
    metadata = row.get("metadata") or {}
    question_id = metadata.get("question_id")
    if question_id is not None:
        return f"question_id:{question_id}"
    question = row.get("seg_question")
    if question is None:
        question = (row.get("query") or {}).get("problem_text")
    if not question:
        raise ValueError(f"Row has no stable query identity: {row.get('example_id')}")
    return f"question_text:{question}"


def build_pairs(
    rows: list[dict[str, Any]],
    seed: int,
    copies_per_donor: int = 1,
    require_full_set: bool = False,
    min_copies_per_donor: int = 1,
    max_donors_per_label: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if copies_per_donor < 1:
        raise ValueError("copies_per_donor must be at least one")
    if not 1 <= min_copies_per_donor <= copies_per_donor:
        raise ValueError(
            "min_copies_per_donor must be between one and copies_per_donor"
        )
    if max_donors_per_label is not None and max_donors_per_label < 1:
        raise ValueError("max_donors_per_label must be positive")
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if label(row) != "NONE":
            pools[family(row)].append(row)

    pairs: list[dict[str, str]] = []
    misses: Counter[str] = Counter()
    recipient_use: Counter[str] = Counter()
    donors = list(rows)
    if max_donors_per_label is not None:
        rng.shuffle(donors)
    accepted_donors: Counter[str] = Counter()
    for donor in donors:
        if label(donor) == "NONE":
            continue
        if (
            max_donors_per_label is not None
            and accepted_donors[label(donor)] >= max_donors_per_label
        ):
            continue
        candidates = list(pools[family(donor)])
        rng.shuffle(candidates)
        valid_candidates: list[tuple[int, dict[str, Any], str]] = []
        donor_query = query_key(donor)
        seen_queries: set[str] = set()
        for tie_break, recipient in enumerate(candidates):
            recipient_query = query_key(recipient)
            if recipient_query == donor_query or recipient_query in seen_queries:
                continue
            is_valid, target_a = valid_pair(donor, recipient)
            if is_valid and target_a is not None:
                valid_candidates.append((tie_break, recipient, target_a))
                seen_queries.add(recipient_query)
        if not valid_candidates:
            misses[label(donor)] += copies_per_donor
            continue

        # The shuffled tie break preserves deterministic diversity while the
        # primary key spreads supervision across recipient queries.  This is
        # intentionally enabled for the multi-query curriculum only; the
        # copies=1 default remains a deterministic one-pair builder.
        if copies_per_donor > 1:
            valid_candidates.sort(
                key=lambda item: (
                    recipient_use[str(item[1]["example_id"])],
                    item[0],
                )
            )
        selected = valid_candidates[:copies_per_donor]
        misses[label(donor)] += copies_per_donor - len(selected)
        if require_full_set and len(selected) != copies_per_donor:
            continue
        if len(selected) < min_copies_per_donor:
            continue
        accepted_donors[label(donor)] += 1
        for _tie_break, recipient, target_a in selected:
            pairs.append(
                {
                    "donor_id": str(donor["example_id"]),
                    "recipient_id": str(recipient["example_id"]),
                    "donor_label": label(donor),
                    "recipient_label": label(recipient),
                    "family": family(donor),
                    "donor_target_on_recipient_query": target_a,
                    "recipient_target": str(recipient["target_label"]),
                    "recipient_query_key": query_key(recipient),
                }
            )
            recipient_use[str(recipient["example_id"])] += 1

    rng.shuffle(pairs)
    donor_use = Counter(pair["donor_id"] for pair in pairs)
    donor_query_counts: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        donor_query_counts[pair["donor_id"]].add(pair["recipient_query_key"])
    stats = {
        "source_rows": len(rows),
        "non_none_donors": sum(label(row) != "NONE" for row in rows),
        "copies_per_donor": copies_per_donor,
        "require_full_set": require_full_set,
        "min_copies_per_donor": min_copies_per_donor,
        "max_donors_per_label": max_donors_per_label,
        "pair_count": len(pairs),
        "unique_donors": len(donor_use),
        "donors_with_full_query_set": sum(
            count == copies_per_donor for count in donor_use.values()
        ),
        "min_distinct_queries_per_paired_donor": min(
            (len(value) for value in donor_query_counts.values()), default=0
        ),
        "max_distinct_queries_per_paired_donor": max(
            (len(value) for value in donor_query_counts.values()), default=0
        ),
        "unique_recipients": len(recipient_use),
        "max_recipient_reuse": max(recipient_use.values(), default=0),
        "misses_by_donor_label": dict(sorted(misses.items())),
        "pairs_by_family": dict(sorted(Counter(pair["family"] for pair in pairs).items())),
        "pairs_by_donor_label": dict(sorted(Counter(pair["donor_label"] for pair in pairs).items())),
    }
    return pairs, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--copies-per-donor", type=int, default=1)
    parser.add_argument("--require-full-set", action="store_true")
    parser.add_argument("--min-copies-per-donor", type=int, default=1)
    parser.add_argument("--max-donors-per-label", type=int)
    args = parser.parse_args()

    pairs, stats = build_pairs(
        read_rows(args.jsonl),
        args.seed,
        args.copies_per_donor,
        args.require_full_set,
        args.min_copies_per_donor,
        args.max_donors_per_label,
    )
    if not pairs:
        raise ValueError("No valid state-swap pairs were built")
    payload = {
        "version": 1,
        "task": TASK,
        "source_jsonl": str(args.jsonl),
        "source_sha256": sha256(args.jsonl),
        "seed": args.seed,
        "stats": stats,
        "pairs": pairs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
