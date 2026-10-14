#!/usr/bin/env python3
"""Build a frozen T6 calibration panel disjoint from every candidate update.

The unified corpus merges the original train and validation rows for the final
fit.  Diagnostic candidates have only consumed a prefix plus explicit index
manifests, so the original-validation rows outside those supports remain valid
selection data.  This builder fails closed on hashes, overlap, and split
identity and writes both the recurrent rows and an audit receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_layerwise_task_retention_manifests import (
    FORMAL_EVAL_SHA256,
    SOURCE_TRAIN_SHA256,
    read_jsonl,
    sha256,
    t6_label,
)


STAGE = "layerwise_t6_calibration_manifest_v1"


def stable_key(row: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row['example_id']}".encode()).hexdigest()


def original_split(row: dict[str, Any]) -> str:
    metadata = row.get("original_metadata") or row.get("metadata") or {}
    explicit = metadata.get("source_split") or metadata.get("split")
    if explicit in {"train", "valid"}:
        return str(explicit)
    example_id = str(row.get("example_id", ""))
    if ":valid:" in example_id:
        return "valid"
    if ":train:" in example_id:
        return "train"
    raise ValueError(f"Cannot recover original split: {example_id}")


def build(
    train_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
    promotion: dict[str, Any],
    excluded_manifests: list[tuple[str, dict[str, Any]]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order = [int(value) for value in promotion.get("real_order") or []]
    prefix_rows = int(promotion.get("completed_prefix_rows", -1))
    if (
        promotion.get("one_epoch_permutation") is not True
        or len(order) != len(train_rows)
        or set(order) != set(range(len(train_rows)))
        or prefix_rows <= 0
    ):
        raise ValueError("Promotion order is not a complete audited permutation")

    excluded = set(order[:prefix_rows])
    exclusion_rows = {"completed_prefix": len(excluded)}
    manifest_receipts = []
    for path, manifest in excluded_manifests:
        indices = {int(value) for value in manifest.get("indices") or []}
        if len(indices) != int(manifest.get("segment_rows", -1)):
            raise ValueError(f"Excluded manifest support drift: {path}")
        if min(indices, default=0) < 0 or max(indices, default=-1) >= len(train_rows):
            raise ValueError(f"Excluded manifest index out of range: {path}")
        exclusion_rows[path] = len(indices)
        excluded.update(indices)
        manifest_receipts.append(
            {
                "path": path,
                "sha256": hashlib.sha256(
                    (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
                ).hexdigest(),
                "rows": len(indices),
            }
        )

    selected_indices = [
        index
        for index, row in enumerate(train_rows)
        if index not in excluded
        and str(row.get("task")) == "predict_error_persistence"
        and original_split(row) == "valid"
    ]
    selected_indices.sort(key=lambda index: stable_key(train_rows[index], seed))
    if len(selected_indices) < 500:
        raise ValueError(f"Insufficient held-out T6 calibration support: {len(selected_indices)}")
    selected = [train_rows[index] for index in selected_indices]
    selected_ids = [str(row["example_id"]) for row in selected]
    formal_ids = {str(row["example_id"]) for row in formal_rows}
    overlap = formal_ids.intersection(selected_ids)
    if overlap:
        raise ValueError(f"Calibration/formal overlap: {sorted(overlap)[:5]}")

    labels = Counter(t6_label(row) for row in selected)
    sources = Counter(str(row.get("source")) for row in selected)
    audit = {
        "stage": STAGE,
        "seed": seed,
        "source_train_sha256": SOURCE_TRAIN_SHA256,
        "formal_eval_sha256": FORMAL_EVAL_SHA256,
        "original_split": "valid",
        "selection_task": "predict_error_persistence",
        "selection_policy": "all_unused_original_validation_rows_stably_ordered",
        "completed_prefix_rows": prefix_rows,
        "excluded_unique_rows": len(excluded),
        "exclusion_rows_before_union": exclusion_rows,
        "excluded_manifests": manifest_receipts,
        "selected_rows": len(selected),
        "selected_index_sha256": hashlib.sha256(
            "\n".join(map(str, selected_indices)).encode()
        ).hexdigest(),
        "selected_example_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode()
        ).hexdigest(),
        "selected_example_id_set_sha256": hashlib.sha256(
            "\n".join(sorted(selected_ids)).encode()
        ).hexdigest(),
        "label_counts": dict(sorted(labels.items())),
        "source_counts": dict(sorted(sources.items())),
        "candidate_training_overlap": len(set(selected_indices).intersection(excluded)),
        "formal_example_id_overlap": len(overlap),
    }
    return selected, audit


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--promotion-order", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if sha256(args.train_jsonl) != SOURCE_TRAIN_SHA256:
        raise ValueError("Calibration source-train SHA256 drift")
    if sha256(args.formal_eval_jsonl) != FORMAL_EVAL_SHA256:
        raise ValueError("Calibration formal-eval SHA256 drift")
    train = read_jsonl(args.train_jsonl)
    formal = read_jsonl(args.formal_eval_jsonl)
    promotion = json.loads(args.promotion_order.read_text(encoding="utf-8"))
    manifests = [
        (str(path), json.loads(path.read_text(encoding="utf-8")))
        for path in args.exclude_manifest
    ]
    rows, audit = build(train, formal, promotion, manifests, seed=args.seed)
    write_jsonl(args.output_jsonl, rows)
    audit["output_jsonl"] = str(args.output_jsonl)
    audit["output_jsonl_sha256"] = sha256(args.output_jsonl)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
