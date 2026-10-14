#!/usr/bin/env python3
"""Build leakage-safe full-history T3 views for the curriculum teacher.

The teacher is deliberately trained and evaluated before it is allowed to
supervise the recurrent student.  It sees the observation-matched direct
history; the eventual student still receives only state tokens plus the query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from state8_natural_curriculum_view import natural_prompt_text, natural_target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def teacher_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("task") != "diagnose_misconception":
        return None
    prompt = natural_prompt_text(
        str(row.get("direct_text") or row.get("full_context_text") or row.get("prompt") or "")
    )
    # The teacher is intentionally full-history and does not use a learned
    # carrier.  Removing the placeholder prevents a pre-trained multimodal
    # token prior from becoming an accidental shortcut in Stage A.
    prompt = prompt.replace("<STUDENT_STATE>", "")
    lowered = prompt.casefold()
    if "<student_history>" not in lowered or "</student_history>" not in lowered:
        raise ValueError(f"Teacher row lacks explicit history: {row.get('example_id')}")
    target = natural_target(row)
    metadata = dict(row.get("metadata") or row.get("original_metadata") or {})
    source = str(
        row.get("source")
        or row.get("dataset")
        or metadata.get("mix_source")
        or metadata.get("eval_source")
        or "unknown"
    )
    protocol = str(metadata.get("eval_protocol") or "teacher_full_history")
    return {
        "example_id": str(row["example_id"]),
        "task": "diagnose_misconception",
        "source": source,
        "protocol": protocol,
        "prompt": prompt,
        "direct_text": prompt,
        "target": target,
        "target_label": row.get("target_label"),
        "metadata": {
            **metadata,
            "eval_source": source,
            "eval_protocol": protocol,
            "teacher_history_visible": True,
            "student_history_visible_at_final_inference": False,
            "output_schema": "label_diagnosis_evidence_v1",
        },
    }


def write_view(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    sources: Counter[str] = Counter()
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
            count += 1
            sources[str(transformed["source"])] += 1
            output.write(json.dumps(transformed, ensure_ascii=False, sort_keys=True) + "\n")
    if count == 0:
        raise ValueError(f"No T3 rows found in {source}")
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(destination),
        "output_sha256": sha256(destination),
        "rows": count,
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
    train_ids = {row["example_id"] for row in read_jsonl(args.output_dir / "train.jsonl")}
    valid_ids = {row["example_id"] for row in read_jsonl(args.output_dir / "valid.jsonl")}
    eval_ids = {row["example_id"] for row in read_jsonl(args.output_dir / "eval.jsonl")}
    overlaps = {
        "train_valid": len(train_ids & valid_ids),
        "train_eval": len(train_ids & eval_ids),
        "valid_eval": len(valid_ids & eval_ids),
    }
    if overlaps["train_eval"] or overlaps["valid_eval"]:
        raise ValueError(f"Teacher train/eval leakage: {overlaps}")
    receipt = {
        "stage": "state8_full_history_teacher_views_v1",
        "teacher_history_visible": True,
        "student_history_visible_at_final_inference": False,
        "output_schema": "label_diagnosis_evidence_v1",
        "outputs": outputs,
        "example_id_overlaps": overlaps,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
