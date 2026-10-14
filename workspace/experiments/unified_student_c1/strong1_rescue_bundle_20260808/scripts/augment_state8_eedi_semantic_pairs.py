#!/usr/bin/env python3
"""Attach train-only canonical-label state pairs to EEDI T3 rows.

The anchor keeps its ordinary natural diagnosis target.  A positive row has
the same canonical misconception label; a hard negative is chosen from the
anchor's frozen candidate labels when possible.  Only updater texts are
attached, so labels never enter the final decoder route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def is_eedi_t3(row: dict[str, Any]) -> bool:
    return row.get("task") == "diagnose_misconception" and row.get("source") == "eedi_mining"


def stable_choice(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No candidates for {key}")
    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def augment(
    train_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus_rows:
        if is_eedi_t3(row) and row.get("update_texts"):
            by_label[str(row["target_label"])].append(row)
    output: list[dict[str, Any]] = []
    counters = Counter()
    for row in train_rows:
        item = dict(row)
        if not is_eedi_t3(row):
            output.append(item)
            continue
        counters["eligible"] += 1
        anchor_id = str(row["example_id"])
        anchor_label = str(row["target_label"])
        positives = [
            candidate
            for candidate in by_label.get(anchor_label, [])
            if str(candidate["example_id"]) != anchor_id
        ]
        if not positives:
            counters["missing_positive"] += 1
            output.append(item)
            continue
        positive = stable_choice(positives, f"{seed}:positive:{anchor_id}")
        hard_labels = [
            str(label)
            for label in row.get("candidate_labels") or []
            if str(label) != anchor_label and by_label.get(str(label))
        ]
        if hard_labels:
            negative_label = stable_choice(
                [{"example_id": label} for label in hard_labels],
                f"{seed}:negative-label:{anchor_id}",
            )["example_id"]
            counters["hard_negative"] += 1
        else:
            negative_labels = [label for label in by_label if label != anchor_label]
            negative_label = stable_choice(
                [{"example_id": label} for label in negative_labels],
                f"{seed}:fallback-label:{anchor_id}",
            )["example_id"]
            counters["fallback_negative"] += 1
        negative = stable_choice(
            by_label[str(negative_label)], f"{seed}:negative-row:{anchor_id}"
        )
        item.update(
            eedi_semantic_pair=True,
            eedi_anchor_label=anchor_label,
            eedi_positive_label=str(positive["target_label"]),
            eedi_negative_label=str(negative["target_label"]),
            eedi_positive_id=str(positive["example_id"]),
            eedi_negative_id=str(negative["example_id"]),
            eedi_anchor_update_texts=list(row["update_texts"]),
            eedi_positive_update_texts=list(positive["update_texts"]),
            eedi_negative_update_texts=list(negative["update_texts"]),
            eedi_pair_supervision_visibility="train_only_not_decoder_input",
        )
        if item["eedi_positive_label"] != anchor_label:
            raise AssertionError("Positive-label drift")
        if item["eedi_negative_label"] == anchor_label:
            raise AssertionError("Negative-label drift")
        counters["paired"] += 1
        output.append(item)
    if counters["eligible"] <= 0 or counters["paired"] <= 0:
        raise ValueError("EEDI semantic-pair augmentation has no usable rows")
    return output, {
        "stage": "state8_eedi_semantic_pair_augmentation_v1",
        "status": "passed",
        "rows": len(output),
        "eligible_eedi_t3": counters["eligible"],
        "paired_eedi_t3": counters["paired"],
        "pair_coverage": counters["paired"] / counters["eligible"],
        "missing_positive": counters["missing_positive"],
        "hard_negative": counters["hard_negative"],
        "fallback_negative": counters["fallback_negative"],
        "canonical_labels_visible_to_decoder": False,
        "pair_supervision_scope": "state_geometry_train_only",
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    rows, receipt = augment(
        read_jsonl(args.train_jsonl), read_jsonl(args.corpus_jsonl), seed=args.seed
    )
    write_jsonl(args.output_jsonl, rows)
    receipt.update(
        train_jsonl=str(args.train_jsonl),
        train_sha256=sha256(args.train_jsonl),
        corpus_jsonl=str(args.corpus_jsonl),
        corpus_sha256=sha256(args.corpus_jsonl),
        output_jsonl=str(args.output_jsonl),
        output_sha256=sha256(args.output_jsonl),
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
