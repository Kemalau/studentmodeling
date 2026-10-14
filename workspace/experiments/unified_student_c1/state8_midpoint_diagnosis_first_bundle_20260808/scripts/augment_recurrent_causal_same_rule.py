#!/usr/bin/env python3
"""Add train-only same-rule native controls to recurrent causal pairs.

The existing specificity-triplet objective already teaches native B, donor A,
and random-state B.  This transform reuses that audited objective without
changing the trainer: every input pair is followed by a second pair whose
native trajectory comes from another student with the recipient's same
malrule.  Both states must therefore preserve recipient answer B on the same
query, while only the different-rule donor should cause answer A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_recurrent_causal_pairs import read_jsonl, state_updates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("malrule_id") or "NONE")


def family(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("family") or "NONE")


def deterministic_control(
    candidates: list[dict[str, Any]], *, seed: int, pair_id: str
) -> dict[str, Any]:
    if not candidates:
        raise ValueError(f"No same-rule control candidates for {pair_id}")
    ordered = sorted(candidates, key=lambda row: str(row["example_id"]))
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def augment(
    source_rows: list[dict[str, Any]],
    causal_rows: list[dict[str, Any]],
    *,
    seed: int,
    max_updates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(row["example_id"]): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("Duplicate example_id in same-rule source")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if label(row) != "NONE":
            buckets[(family(row), label(row))].append(row)

    output: list[dict[str, Any]] = []
    controls: list[str] = []
    for index, row in enumerate(causal_rows):
        donor_id = str(row["donor_id"])
        recipient_id = str(row["recipient_id"])
        recipient_label = str(row["recipient_label"])
        pair_family = str(row["family"])
        if donor_id not in by_id or recipient_id not in by_id:
            raise ValueError(f"Causal pair references unknown source row: {row.get('pair_id')}")
        if label(by_id[recipient_id]) != recipient_label:
            raise ValueError(f"Recipient-label drift: {row.get('pair_id')}")
        if family(by_id[recipient_id]) != pair_family:
            raise ValueError(f"Recipient-family drift: {row.get('pair_id')}")
        candidates = [
            candidate
            for candidate in buckets[(pair_family, recipient_label)]
            if str(candidate["example_id"]) not in {recipient_id, donor_id}
        ]
        control = deterministic_control(
            candidates, seed=seed, pair_id=str(row["pair_id"])
        )
        control_id = str(control["example_id"])
        controls.append(control_id)
        recipient_updates = list(
            row.get("native_update_texts")
            or state_updates(by_id[recipient_id], max_updates)
        )
        control_updates = state_updates(control, max_updates)

        native = dict(row)
        native.update(
            pair_id=f"recurrent-causal-samerule:{2 * index:07d}",
            native_control_role="recipient_native",
            native_state_source_id=recipient_id,
            same_rule_control_id=control_id,
            same_rule_recipient_id=recipient_id,
            recipient_native_update_texts=recipient_updates,
            same_rule_control_update_texts=control_updates,
        )
        same_rule = dict(row)
        same_rule.update(
            pair_id=f"recurrent-causal-samerule:{2 * index + 1:07d}",
            native_update_texts=control_updates,
            native_control_role="same_rule_native",
            native_state_source_id=control_id,
            same_rule_control_id=control_id,
            same_rule_recipient_id=recipient_id,
            recipient_native_update_texts=recipient_updates,
            same_rule_control_update_texts=control_updates,
        )
        output.extend((native, same_rule))

    if not output:
        raise ValueError("Cannot augment an empty causal split")
    stats = {
        "input_pairs": len(causal_rows),
        "output_pairs": len(output),
        "recipient_native_pairs": len(causal_rows),
        "same_rule_native_pairs": len(causal_rows),
        "unique_same_rule_controls": len(set(controls)),
        "control_reuse": dict(sorted(Counter(controls).items())),
        "raw_history_visible_to_decoder": False,
        "objective_alignment": "swap_vs_random_plus_same_rule_control",
    }
    return output, stats


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def transform_split(
    source: Path,
    causal: Path,
    output: Path,
    *,
    seed: int,
    max_updates: int,
) -> dict[str, Any]:
    source_rows = list(read_jsonl(source))
    causal_rows = list(read_jsonl(causal))
    rows, stats = augment(
        source_rows, causal_rows, seed=seed, max_updates=max_updates
    )
    write_jsonl(output, rows)
    return {
        **stats,
        "source": str(source),
        "source_sha256": sha256(source),
        "causal_input": str(causal),
        "causal_input_sha256": sha256(causal),
        "output": str(output),
        "output_sha256": sha256(output),
        "seed": seed,
        "max_updates": max_updates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--valid-source", type=Path, required=True)
    parser.add_argument("--causal-train", type=Path, required=True)
    parser.add_argument("--causal-valid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-updates", type=int, default=4)
    args = parser.parse_args()
    if args.max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    train = transform_split(
        args.train_source,
        args.causal_train,
        args.output_dir / "train.jsonl",
        seed=args.seed,
        max_updates=args.max_updates,
    )
    valid = transform_split(
        args.valid_source,
        args.causal_valid,
        args.output_dir / "valid.jsonl",
        seed=args.seed + 1,
        max_updates=args.max_updates,
    )
    receipt = {
        "stage": "recurrent_causal_same_rule_augmentation_v1",
        "status": "passed",
        "train": train,
        "valid": valid,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "augmentation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
