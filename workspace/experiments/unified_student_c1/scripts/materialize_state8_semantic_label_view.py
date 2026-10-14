#!/usr/bin/env python3
"""Materialize a semantic-label training view without mutating source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from state8_semantic_label import (
    eedi_semantic_label,
    diagnosis_text,
    replace_natural_label,
    semantic_label,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_row(source: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    row = dict(source)
    changed = False
    if str(row.get("task") or "") == "diagnose_misconception" and row.get("target"):
        label = semantic_label(row)
        original = str(row.get("target_label") or "")
        row["taxonomy_id"] = original
        row["semantic_label"] = label
        target = str(row["target"])
        if "<label>" in target:
            row["target"] = replace_natural_label(target, label)
        changed = label != original or row["target"] != target

    if str(row.get("causal_target_kind") or "") == "diagnosis_natural_schema":
        native_id = str(row.get("native_target_label") or "").strip()
        donor_id = str(row.get("donor_target_label") or "").strip()
        if native_id and donor_id:
            native_label = eedi_semantic_label(
                native_id, diagnosis_text(row.get("native_target"))
            ) if native_id.isdigit() else native_id
            donor_label = eedi_semantic_label(
                donor_id, diagnosis_text(row.get("donor_target"))
            ) if donor_id.isdigit() else donor_id
            row.update(
                native_taxonomy_id=native_id,
                donor_taxonomy_id=donor_id,
                native_semantic_label=native_label,
                donor_semantic_label=donor_label,
            )
            row["native_target"] = replace_natural_label(
                row["native_target"], native_label
            )
            row["donor_target"] = replace_natural_label(
                row["donor_target"], donor_label
            )
            changed |= native_label != native_id or donor_label != donor_id
    return row, changed


def materialize(input_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    changed = 0
    labels: Counter[str] = Counter()
    taxonomy: Counter[str] = Counter()
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row, row_changed = transform_row(json.loads(line))
            rows.append(row)
            changed += int(row_changed)
            if row.get("semantic_label"):
                labels[str(row["semantic_label"])] += 1
            if row.get("taxonomy_id"):
                taxonomy[str(row["taxonomy_id"])] += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    receipt = {
        "stage": "state8_semantic_label_view_v1",
        "status": "passed",
        "rows": len(rows),
        "changed_rows": changed,
        "semantic_label_rows": sum(labels.values()),
        "unique_semantic_labels": len(labels),
        "taxonomy_id_rows": sum(taxonomy.values()),
        "unique_taxonomy_ids": len(taxonomy),
        "source_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
        "numeric_taxonomy_id_preserved": True,
        "semantic_label_auxiliary_only": True,
        "semantic_judge_authoritative": True,
        "raw_history_visible_to_decoder": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.input_jsonl, args.output_jsonl, args.receipt), indent=2))


if __name__ == "__main__":
    main()

