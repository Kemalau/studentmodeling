#!/usr/bin/env python3
"""Build a fresh controlled repair segment after promoted round 1250."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from build_layerwise_task_retention_manifests import (
    ANCHOR_SPECS,
    FORMAL_EVAL_SHA256,
    SOURCE_TRAIN_SHA256,
    choose,
    read_jsonl,
    sha256,
    t6_label,
)


START_GLOBAL_POSITION = 20_000
SEGMENT_ROWS = 1_000
TASK_COUNTS = {
    "diagnose_misconception": 200,
    "predict_correctness": 80,
    "predict_error_persistence": 600,
    "predict_prospective_error": 70,
    "predict_response": 50,
}


def build(
    train_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    if checkpoint.get("distributed") is not True:
        raise ValueError("Source checkpoint is not distributed")
    if int(checkpoint.get("round", -1)) != 1250:
        raise ValueError("Source checkpoint round drift")
    if int(checkpoint.get("world_size", -1)) != 16:
        raise ValueError("Source checkpoint world-size drift")
    real_order = checkpoint.get("real_order")
    if not isinstance(real_order, list) or len(real_order) != len(train_rows):
        raise ValueError("Source checkpoint real-order support drift")
    seen_prefix = {int(index) for index in real_order[:START_GLOBAL_POSITION]}
    if len(seen_prefix) != START_GLOBAL_POSITION:
        raise ValueError("Source checkpoint prefix is not unique")

    by_task_source: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_t6_source_label: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        task, source = str(row["task"]), str(row["source"])
        by_task_source[(task, source)].append(index)
        if task == "predict_error_persistence":
            by_t6_source_label[(source, t6_label(row))].append(index)

    used = set(seen_prefix)
    selected: list[int] = []
    for key, count in ANCHOR_SPECS.items():
        selected.extend(choose(by_task_source, key, count, seed=seed, used=used))
    selected.extend(
        choose(
            by_task_source,
            ("diagnose_misconception", "malrule"),
            200,
            seed=seed + 101,
            used=used,
        )
    )
    for label_name in ("persistent", "transient"):
        selected.extend(
            choose(
                by_t6_source_label,
                ("foundationalassist", label_name),
                300,
                seed=seed + 211,
                used=used,
            )
        )
    if len(selected) != SEGMENT_ROWS or len(set(selected)) != SEGMENT_ROWS:
        raise AssertionError((len(selected), len(set(selected))))
    if seen_prefix.intersection(selected):
        raise AssertionError("Repair segment overlaps promoted round-1250 prefix")
    random.Random(f"{seed}:promoted-round1250-same-rule-repair").shuffle(selected)

    formal_ids = {str(row["example_id"]) for row in formal_rows}
    selected_ids = [str(train_rows[index]["example_id"]) for index in selected]
    if formal_ids.intersection(selected_ids):
        raise ValueError("Repair segment overlaps formal evaluation")
    task_counts = Counter(str(train_rows[index]["task"]) for index in selected)
    if dict(sorted(task_counts.items())) != TASK_COUNTS:
        raise AssertionError(task_counts)
    source_counts = Counter(str(train_rows[index]["source"]) for index in selected)
    t6_counts = Counter(
        t6_label(train_rows[index])
        for index in selected
        if train_rows[index]["task"] == "predict_error_persistence"
    )
    return {
        "stage": "layerwise_task_retention_index_manifest_v1",
        "branch": "eedi_full",
        "seed": seed,
        "source_train_sha256": SOURCE_TRAIN_SHA256,
        "formal_eval_sha256": FORMAL_EVAL_SHA256,
        "formal_example_id_overlap": 0,
        "source_checkpoint_round": 1250,
        "source_checkpoint_world_size": 16,
        "seen_prefix_rows": START_GLOBAL_POSITION,
        "seen_prefix_overlap": 0,
        "start_global_position": START_GLOBAL_POSITION,
        "segment_rows": SEGMENT_ROWS,
        "selected_example_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "t6_label_counts": dict(sorted(t6_counts.items())),
        "indices": selected,
        "curriculum_mode": "fresh_balanced_t3_t6_repair_after_promoted_round1250",
        "controlled_factor": "causal_same_rule_augmentation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if sha256(args.train_jsonl) != SOURCE_TRAIN_SHA256:
        raise ValueError("Repair source-train SHA256 drift")
    if sha256(args.formal_eval_jsonl) != FORMAL_EVAL_SHA256:
        raise ValueError("Repair formal-eval SHA256 drift")
    train_rows = read_jsonl(args.train_jsonl)
    formal_rows = read_jsonl(args.formal_eval_jsonl)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    value = build(train_rows, formal_rows, checkpoint, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value[key] for key in (
        "stage", "start_global_position", "segment_rows", "seen_prefix_overlap",
        "formal_example_id_overlap", "task_counts", "source_counts", "t6_label_counts",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
