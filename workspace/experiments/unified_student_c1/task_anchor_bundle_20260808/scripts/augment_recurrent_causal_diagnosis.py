#!/usr/bin/env python3
"""Add natural-language diagnosis twins to answer-level recurrent C4 pairs.

Each existing pair keeps its answer-prediction objective.  A second row reuses
the exact native/donor state trajectories and identical recipient query, but
requires the closed decoder to emit the corresponding canonical label,
natural-language diagnosis, and state-grounded evidence.  This makes
diagnosis-level intervention an explicit training objective instead of an
inference-only extrapolation from answer C4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from state8_natural_curriculum_view import natural_target
    from t3_judge_utils import natural_schema_valid
except ImportError:  # package-style unit tests
    from scripts.state8_natural_curriculum_view import natural_target
    from scripts.t3_judge_utils import natural_schema_valid


QUERY = re.compile(r"<query>\s*(.*?)\s*</query>", re.I | re.S)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_query(decode_text: str) -> str:
    match = QUERY.search(str(decode_text))
    if not match:
        raise ValueError("Causal decode text lacks one <query> block")
    body = match.group(1).strip()
    body = re.split(r"(?im)^\s*Valid outputs:\s*$", body, maxsplit=1)[0].strip()
    if not body:
        raise ValueError("Causal decode query is empty after removing valid outputs")
    return f"<query>\n{body}\n</query>"


def diagnosis_decode_text(row: dict[str, Any]) -> str:
    return (
        "You are modeling one student's mathematical behavior. Diagnose the "
        "specific systematic misconception encoded in the student's latent "
        "state. Return exactly one non-empty <label>, <diagnosis>, and "
        "<evidence> tag.\n"
        "<STUDENT_STATE>\n"
        f"{clean_query(str(row['decode_text']))}\n"
        "Answer:"
    )


def target_for(label: str, update_texts: list[str]) -> str:
    target = natural_target(
        {
            "target": f"<misconception>{label}</misconception>",
            "target_label": label,
            "update_texts": list(update_texts),
        }
    )
    if not natural_schema_valid(target):
        raise AssertionError(f"Invalid natural diagnosis target for {label}")
    return target


def diagnosis_twin(row: dict[str, Any]) -> dict[str, Any]:
    native_label = str(row["recipient_label"])
    donor_label = str(row["donor_label"])
    if native_label == donor_label:
        raise ValueError(f"Diagnosis twin has identical labels: {native_label}")
    native_target = target_for(native_label, list(row["native_update_texts"]))
    donor_target = target_for(donor_label, list(row["donor_update_texts"]))
    item = dict(row)
    item.update(
        pair_id=f"{row['pair_id']}:diagnosis",
        paired_answer_pair_id=str(row["pair_id"]),
        causal_target_kind="diagnosis_natural_schema",
        task="diagnose_misconception",
        decode_text=diagnosis_decode_text(row),
        native_target=native_target,
        donor_target=donor_target,
        native_target_label=native_label,
        donor_target_label=donor_label,
        candidate_labels=[native_label, donor_label],
        candidate_targets={native_label: native_target, donor_label: donor_target},
        output_schema="label_diagnosis_evidence_v1",
        canonical_label_auxiliary_supervision=True,
        raw_history_visible_to_decoder=False,
    )
    return item


def augment(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot augment an empty causal split")
    output: list[dict[str, Any]] = []
    for row in rows:
        answer = dict(row)
        answer.setdefault("causal_target_kind", "prospective_answer")
        output.extend((answer, diagnosis_twin(row)))
    pair_ids = [str(row["pair_id"]) for row in output]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Diagnosis augmentation produced duplicate pair IDs")
    counts = Counter(str(row["causal_target_kind"]) for row in output)
    expected = Counter({"prospective_answer": len(rows), "diagnosis_natural_schema": len(rows)})
    if counts != expected:
        raise AssertionError((counts, expected))
    return output, {
        "input_pairs": len(rows),
        "output_pairs": len(output),
        "target_kind_counts": dict(sorted(counts.items())),
        "canonical_label_auxiliary_supervision": True,
        "diagnosis_output_schema": "label_diagnosis_evidence_v1",
        "raw_history_visible_to_decoder": False,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def transform(source: Path, output: Path) -> dict[str, Any]:
    rows, stats = augment(read_jsonl(source))
    write_jsonl(output, rows)
    return {
        **stats,
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
    train = transform(args.train_jsonl, args.output_dir / "train.jsonl")
    valid = transform(args.valid_jsonl, args.output_dir / "valid.jsonl")
    receipt = {
        "stage": "recurrent_causal_answer_diagnosis_twins_v1",
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
