#!/usr/bin/env python3
"""Build recurrent donor-state/recipient-query causal training pairs.

Each record has two independently updated states and one identical recipient
query.  The recipient state is supervised with its native response B, while
the donor state is supervised with the executable donor response A on that
same query.  Raw histories never enter the decoder view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from recurrent_student_model import STATE_TOKEN, build_update_text, extract_direct_decode_text


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


def read_source(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        example_id = str(row["example_id"])
        if example_id in rows:
            raise ValueError(f"Duplicate source example: {example_id}")
        rows[example_id] = row
    return rows


def closed_decode_text(recipient: dict[str, Any]) -> str:
    text = extract_direct_decode_text(str(recipient["prompt"]))
    if "<student_history>" in text.lower():
        raise AssertionError("Recipient decoder text leaked raw history")
    if text.count(STATE_TOKEN) != 1:
        raise AssertionError("Recipient decoder must have exactly one state carrier")
    return text


def direct_text(donor: dict[str, Any], recipient: dict[str, Any]) -> str:
    required = ("seg_instruction", "seg_history", "seg_question")
    if any(key not in donor for key in required[:2]) or required[2] not in recipient:
        raise ValueError("Causal pair source rows lack segmented prompt fields")
    text = (
        str(donor["seg_instruction"])
        + str(donor["seg_history"])
        + STATE_TOKEN
        + "\n"
        + str(recipient["seg_question"])
    )
    if text.count(STATE_TOKEN) != 1:
        raise AssertionError("Direct causal comparator must have one state carrier")
    return text


def state_updates(row: dict[str, Any], max_updates: int) -> list[str]:
    history = list(row.get("history") or [])
    if not history:
        raise ValueError(f"Causal state row has no observed history: {row.get('example_id')}")
    selected = history[-max_updates:]
    updates = [build_update_text(interaction) for interaction in selected]
    if any(update.count(STATE_TOKEN) != 2 for update in updates):
        raise AssertionError("Each recurrent update must contain exactly two carriers")
    if any("<student_history>" in update.lower() for update in updates):
        raise AssertionError("Recurrent update leaked a raw history container")
    return updates


def build_pair(
    pair_index: int,
    spec: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    max_updates: int,
) -> dict[str, Any]:
    donor_id = str(spec["donor_id"])
    recipient_id = str(spec["recipient_id"])
    if donor_id == recipient_id:
        raise ValueError(f"Self-pair is not causal: {donor_id}")
    donor = rows[donor_id]
    recipient = rows[recipient_id]
    donor_label = str(spec["donor_target_on_recipient_query"])
    native_label = str(spec["recipient_target"])
    candidates = {str(key): str(value) for key, value in recipient["candidate_targets"].items()}
    if donor_label not in candidates or native_label not in candidates:
        raise ValueError(
            f"Pair targets are not recipient candidates: donor={donor_label!r}, "
            f"native={native_label!r}"
        )
    if donor_label == native_label:
        raise ValueError(f"Pair has no answer intervention: {donor_id} -> {recipient_id}")
    if str(recipient["target_label"]) != native_label:
        raise ValueError(f"Recipient native target drift for {recipient_id}")
    decode = closed_decode_text(recipient)
    donor_direct = direct_text(donor, recipient)
    native_direct = str(recipient["prompt"])
    if native_direct.count(STATE_TOKEN) != 1:
        raise AssertionError("Native direct comparator must have one state carrier")
    return {
        "pair_id": f"recurrent-causal:{pair_index:07d}",
        "donor_id": donor_id,
        "recipient_id": recipient_id,
        "donor_label": str(spec["donor_label"]),
        "recipient_label": str(spec["recipient_label"]),
        "family": str(spec["family"]),
        "donor_update_texts": state_updates(donor, max_updates),
        "native_update_texts": state_updates(recipient, max_updates),
        "decode_text": decode,
        "donor_direct_text": donor_direct,
        "native_direct_text": native_direct,
        "donor_target_label": donor_label,
        "native_target_label": native_label,
        "donor_target": candidates[donor_label],
        "native_target": candidates[native_label],
        "candidate_labels": list(recipient["candidate_labels"]),
        "candidate_targets": candidates,
        "information_route": "recurrent_donor_or_native_state_plus_identical_recipient_query",
        "raw_history_visible_to_decoder": False,
    }


def build_split(
    source_jsonl: Path,
    pair_manifest: Path,
    output_jsonl: Path,
    max_updates: int,
) -> dict[str, Any]:
    payload = json.loads(pair_manifest.read_text(encoding="utf-8"))
    if payload.get("task") != "malrule_next_response":
        raise ValueError(f"Unexpected causal task: {payload.get('task')!r}")
    source_hash = sha256(source_jsonl)
    if payload.get("source_sha256") != source_hash:
        raise ValueError("Causal pair manifest source SHA256 mismatch")
    rows = read_source(source_jsonl)
    records = [
        build_pair(index, spec, rows, max_updates)
        for index, spec in enumerate(payload.get("pairs") or [])
    ]
    if not records:
        raise ValueError("Causal pair manifest is empty")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    families = Counter(record["family"] for record in records)
    return {
        "source_jsonl": str(source_jsonl),
        "source_sha256": source_hash,
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": sha256(pair_manifest),
        "output_jsonl": str(output_jsonl),
        "output_sha256": sha256(output_jsonl),
        "pairs": len(records),
        "max_updates": max_updates,
        "families": dict(sorted(families.items())),
        "decoder_history_leaks": 0,
        "answer_intervention_errors": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--valid-source", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--valid-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=4)
    args = parser.parse_args()
    if args.max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    train = build_split(
        args.train_source,
        args.train_pairs,
        args.output_dir / "train.jsonl",
        args.max_updates,
    )
    valid = build_split(
        args.valid_source,
        args.valid_pairs,
        args.output_dir / "valid.jsonl",
        args.max_updates,
    )
    stats = {"version": 1, "status": "passed", "train": train, "valid": valid}
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
