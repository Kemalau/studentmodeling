#!/usr/bin/env python3
"""Freeze a balanced native/same-rule causal subset for a short curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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


def twin_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row["same_rule_recipient_id"]),
        str(row["same_rule_control_id"]),
        str(row["donor_id"]),
    )


def select(rows: list[dict], twins: int, seed: int) -> list[dict]:
    if twins <= 0:
        raise ValueError("twins must be positive")
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(twin_key(row), []).append(row)
    malformed = {
        key: Counter(str(row.get("native_control_role")) for row in group)
        for key, group in grouped.items()
        if Counter(str(row.get("native_control_role")) for row in group)
        != Counter({"recipient_native": 1, "same_rule_native": 1})
    }
    if malformed:
        raise ValueError(f"Malformed causal twins: {list(malformed.items())[:2]}")
    ordered = sorted(
        grouped,
        key=lambda key: hashlib.sha256(
            f"{seed}:{key[0]}:{key[1]}:{key[2]}".encode()
        ).hexdigest(),
    )
    if len(ordered) < twins:
        raise ValueError(f"Need {twins} twins, found {len(ordered)}")
    output: list[dict] = []
    for index, key in enumerate(ordered[:twins]):
        group = sorted(
            grouped[key],
            key=lambda row: 0 if row["native_control_role"] == "recipient_native" else 1,
        )
        for offset, row in enumerate(group):
            item = dict(row)
            item["pair_id"] = f"state8-qualified-causal:{2 * index + offset:07d}"
            output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--twins", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = select(read(args.input_jsonl), args.twins, args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    roles = Counter(str(row["native_control_role"]) for row in rows)
    families = Counter(str(row["family"]) for row in rows)
    if roles != Counter({"recipient_native": args.twins, "same_rule_native": args.twins}):
        raise AssertionError(roles)
    receipt = {
        "stage": "state8_qualified_causal_twin_subset_v1",
        "input_sha256": sha256(args.input_jsonl),
        "output_sha256": sha256(args.output_jsonl),
        "seed": args.seed,
        "twins": args.twins,
        "pairs": len(rows),
        "role_counts": dict(sorted(roles.items())),
        "family_counts": dict(sorted(families.items())),
    }
    args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
