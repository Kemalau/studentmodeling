#!/usr/bin/env python3
"""Build response-only state-formation data with counterfactual query recombination.

For each original Malrule response row, retain the original example and create
one or more examples that pair the same student history with a different query
from the same mathematical family.  The target is the answer produced by the
history student's malrule on that new query, using the executable-answer map
already frozen in the cleaned Malrule dataset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATE_TOKEN = "<STUDENT_STATE>"
TASK = "malrule_next_response"


def read_response_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") == TASK:
                rows.append(row)
    if not rows:
        raise ValueError(f"No {TASK} rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def malrule_id(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("malrule_id", ""))


def family(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("family", ""))


def raw_id(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("raw_example_id", row.get("example_id", "")))


def counterfactual_target(donor: dict[str, Any], recipient: dict[str, Any]) -> str | None:
    donor_rule = malrule_id(donor)
    if donor_rule == "NONE":
        answer = recipient.get("query", {}).get("correct_answer")
    else:
        answer = recipient.get("metadata", {}).get("query_answers_by_malrule", {}).get(donor_rule)
    if answer is None:
        return None
    answer = str(answer)
    if donor_rule != "NONE" and answer == str(recipient.get("query", {}).get("correct_answer", "")):
        return None
    if answer not in recipient.get("candidate_targets", {}):
        return None
    return answer


def make_counterfactual(
    donor: dict[str, Any],
    recipient: dict[str, Any],
    target_label: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    row = copy.deepcopy(donor)
    row["dataset"] = "malrulelib_clean_state_curriculum"
    row["example_id"] = f"state-curriculum:{split}:{index:07d}"
    row["query"] = copy.deepcopy(recipient["query"])
    row["query"]["student_answer"] = target_label
    row["query"]["malrule_answer"] = None if malrule_id(donor) == "NONE" else target_label
    row["query"]["is_correct"] = target_label == str(row["query"].get("correct_answer", ""))
    row["candidate_labels"] = copy.deepcopy(recipient["candidate_labels"])
    row["candidate_targets"] = copy.deepcopy(recipient["candidate_targets"])
    row["target_label"] = target_label
    row["target"] = row["candidate_targets"][target_label]
    row["seg_question"] = recipient["seg_question"]
    row["prompt"] = f"{row['seg_instruction']}{row['seg_history']}{STATE_TOKEN}\n{row['seg_question']}"
    row["metadata"] = copy.deepcopy(donor["metadata"])
    row["metadata"].update(
        {
            "counterfactual_recombined": True,
            "history_source_example_id": donor["example_id"],
            "query_source_example_id": recipient["example_id"],
            "question_id": recipient.get("metadata", {}).get("question_id"),
            "query_answers_by_malrule": copy.deepcopy(
                recipient.get("metadata", {}).get("query_answers_by_malrule", {})
            ),
        }
    )
    return row


def build_split(
    rows: list[dict[str, Any]], split: str, seed: int, copies_per_original: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[family(row)].append(row)

    originals = [copy.deepcopy(row) for row in rows]
    for row in originals:
        row.setdefault("metadata", {})["counterfactual_recombined"] = False

    counterfactuals: list[dict[str, Any]] = []
    misses: Counter[str] = Counter()
    for donor in rows:
        candidates = list(pools[family(donor)])
        rng.shuffle(candidates)
        used_queries: set[str] = set()
        made = 0
        for recipient in candidates:
            if raw_id(recipient) == raw_id(donor) or raw_id(recipient) in used_queries:
                continue
            target = counterfactual_target(donor, recipient)
            if target is None:
                continue
            counterfactuals.append(
                make_counterfactual(donor, recipient, target, split, len(counterfactuals))
            )
            used_queries.add(raw_id(recipient))
            made += 1
            if made >= copies_per_original:
                break
        if made < copies_per_original:
            misses[malrule_id(donor)] += copies_per_original - made

    combined = originals + counterfactuals
    rng.shuffle(combined)
    stats = {
        "split": split,
        "seed": seed,
        "copies_per_original": copies_per_original,
        "original_rows": len(originals),
        "counterfactual_rows": len(counterfactuals),
        "total_rows": len(combined),
        "counterfactual_misses_by_malrule": dict(sorted(misses.items())),
        "family_counts": dict(sorted(Counter(family(row) for row in combined).items())),
        "malrule_counts": dict(sorted(Counter(malrule_id(row) for row in combined).items())),
    }
    return combined, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copies-per-original", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    if args.copies_per_original < 1:
        raise ValueError("--copies-per-original must be >= 1")

    train, train_stats = build_split(
        read_response_rows(args.train_input), "train", args.seed, args.copies_per_original
    )
    valid, valid_stats = build_split(
        read_response_rows(args.valid_input), "valid", args.seed + 1, args.copies_per_original
    )
    train_path = args.output_dir / "train.jsonl"
    valid_path = args.output_dir / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    stats = {
        "source_train": str(args.train_input),
        "source_valid": str(args.valid_input),
        "train": train_stats,
        "valid": valid_stats,
        "sha256": {"train": sha256(train_path), "valid": sha256(valid_path)},
    }
    stats_path = args.output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
