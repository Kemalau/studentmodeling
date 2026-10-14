#!/usr/bin/env python3
"""Build leakage-safe full-history views with diagnosis-first supervision.

The qualified teacher and recurrent student must use the same output schema.
This builder keeps the canonical label as auxiliary supervision, but places
the natural-language diagnosis and evidence first:

    <diagnosis>...</diagnosis>
    <evidence>...</evidence>
    <label>...</label>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_state8_teacher_views import (
    read_jsonl,
    sha256,
    teacher_row as base_teacher_row,
    write_view as _unused_write_view,
)
from train_state8_diagnosis_first_causal_repair import (
    FIELD_ORDER,
    diagnosis_first_prompt,
    reorder_target,
)


def teacher_row(row: dict[str, Any]) -> dict[str, Any] | None:
    transformed = base_teacher_row(row)
    if transformed is None:
        return None
    transformed["target"] = reorder_target(str(transformed["target"]))
    prompt = diagnosis_first_prompt(str(transformed["prompt"]))
    transformed["prompt"] = prompt
    transformed["direct_text"] = prompt
    metadata = dict(transformed.get("metadata") or {})
    metadata.update(
        output_schema="diagnosis_evidence_label_v1",
        diagnosis_field_order=list(FIELD_ORDER),
        semantic_fields_primary=True,
        canonical_label_role="auxiliary_supervision_and_reporting",
    )
    transformed["metadata"] = metadata
    return transformed


def write_view(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    sources: dict[str, int] = {}
    seen: set[str] = set()
    with destination.open("w", encoding="utf-8") as output:
        for row in read_jsonl(source):
            transformed = teacher_row(row)
            if transformed is None:
                continue
            example_id = str(transformed["example_id"])
            if example_id in seen:
                raise ValueError(f"Duplicate example_id in teacher view: {example_id}")
            seen.add(example_id)
            rows += 1
            source_name = str(transformed["source"])
            sources[source_name] = sources.get(source_name, 0) + 1
            output.write(
                json.dumps(transformed, ensure_ascii=False, sort_keys=True) + "\n"
            )
    if rows == 0:
        raise ValueError(f"No T3 rows found in {source}")
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(destination),
        "output_sha256": sha256(destination),
        "rows": rows,
        "source_counts": dict(sorted(sources.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = {
        name: write_view(path, args.output_dir / f"{name}.jsonl")
        for name, path in (
            ("train", args.train_jsonl),
            ("valid", args.valid_jsonl),
            ("eval", args.eval_jsonl),
        )
    }
    id_sets = {
        name: {
            str(row["example_id"])
            for row in read_jsonl(args.output_dir / f"{name}.jsonl")
        }
        for name in ("train", "valid", "eval")
    }
    overlaps = {
        "train_valid": len(id_sets["train"] & id_sets["valid"]),
        "train_eval": len(id_sets["train"] & id_sets["eval"]),
        "valid_eval": len(id_sets["valid"] & id_sets["eval"]),
    }
    if overlaps["train_eval"] or overlaps["valid_eval"]:
        raise ValueError(f"Teacher train/eval leakage: {overlaps}")
    receipt = {
        "stage": "state8_diagnosis_first_full_history_teacher_views_v1",
        "teacher_history_visible": True,
        "student_history_visible_at_final_inference": False,
        "output_schema": "diagnosis_evidence_label_v1",
        "diagnosis_field_order": list(FIELD_ORDER),
        "semantic_fields_primary": True,
        "canonical_label_role": "auxiliary_supervision_and_reporting",
        "outputs": outputs,
        "example_id_overlaps": overlaps,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
