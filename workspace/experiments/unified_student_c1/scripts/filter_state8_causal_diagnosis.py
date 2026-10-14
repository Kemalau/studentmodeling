#!/usr/bin/env python3
"""Materialize diagnosis-only C4 pairs for high-exposure State8 consolidation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if str(row.get("causal_target_kind")) == "diagnosis_natural_schema"
    ]
    if not selected:
        raise ValueError("No diagnosis-natural-schema causal rows were found")
    roles = Counter(str(row.get("native_control_role")) for row in selected)
    expected = Counter(
        {
            "recipient_native": len(selected) // 2,
            "same_rule_native": len(selected) // 2,
        }
    )
    if len(selected) % 2 or roles != expected:
        raise ValueError(f"Diagnosis causal role imbalance: {roles}")
    if len({str(row["pair_id"]) for row in selected}) != len(selected):
        raise ValueError("Diagnosis-only causal rows contain duplicate pair IDs")
    if any(bool(row.get("raw_history_visible_to_decoder")) for row in selected):
        raise ValueError("Diagnosis-only causal decoder must not see raw history")
    return selected, {
        "rows": len(selected),
        "role_counts": dict(sorted(roles.items())),
        "target_kind": "diagnosis_natural_schema",
        "raw_history_visible_to_decoder": False,
    }


def transform(source: Path, output: Path) -> dict[str, Any]:
    selected, audit = filter_rows(read_jsonl(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)
    return {
        **audit,
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = {
        "stage": "state8_causal_diagnosis_only_v1",
        "status": "passed",
        "train": transform(args.train_jsonl, args.output_dir / "train.jsonl"),
        "valid": transform(args.valid_jsonl, args.output_dir / "valid.jsonl"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
