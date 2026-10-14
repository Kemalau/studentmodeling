#!/usr/bin/env python3
"""Build fresh T6 repair curricula after the 5000-row open-T3 formation arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_layerwise_task_retention_manifests import (
    ANCHOR_SPECS,
    FORMAL_EVAL_SHA256,
    SOURCE_TRAIN_SHA256,
    choose,
    read_jsonl,
    sha256,
    t6_label,
)


PRIOR_MANIFEST_SHA256 = (
    "97bef6748b1fa8d75809e9e01c9b76d57b6534d336b7037217289932a31df393"
)
START_GLOBAL_POSITION = 17_000
SEGMENT_ROWS = 1_000
SPECS = {
    "balanced": {
        # Formation already consumes every Eedi row. Preserve open-T3 behavior
        # with fresh Malrule diagnosis rehearsal instead of silently repeating
        # Eedi examples from the prior segment.
        "t3": {"malrule": 200},
        "t6": {
            ("foundationalassist", "persistent"): 300,
            ("foundationalassist", "transient"): 300,
        },
    },
    "strong": {
        "t3": {"malrule": 100},
        "t6": {
            ("foundationalassist", "persistent"): 350,
            ("foundationalassist", "transient"): 350,
        },
    },
}


def build(
    train_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
    prior: dict[str, Any],
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    if variant not in SPECS:
        raise ValueError(f"Unknown repair variant: {variant}")
    prior_indices = {int(value) for value in prior["indices"]}
    if (
        int(prior.get("start_global_position", -1)) != 12_000
        or int(prior.get("segment_rows", -1)) != 5_000
        or len(prior_indices) != 5_000
    ):
        raise ValueError("Prior open-T3 formation support drift")

    by_task_source: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_t6_source_label: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        task, source = str(row["task"]), str(row["source"])
        by_task_source[(task, source)].append(index)
        if task == "predict_error_persistence":
            by_t6_source_label[(source, t6_label(row))].append(index)

    used = set(prior_indices)
    selected: list[int] = []
    for key, count in ANCHOR_SPECS.items():
        selected.extend(choose(by_task_source, key, count, seed=seed, used=used))
    spec = SPECS[variant]
    for source, count in spec["t3"].items():
        selected.extend(
            choose(
                by_task_source,
                ("diagnose_misconception", source),
                count,
                seed=seed + 101,
                used=used,
            )
        )
    for key, count in spec["t6"].items():
        selected.extend(
            choose(by_t6_source_label, key, count, seed=seed + 211, used=used)
        )
    if len(selected) != SEGMENT_ROWS or len(set(selected)) != SEGMENT_ROWS:
        raise AssertionError((variant, len(selected), len(set(selected))))
    if prior_indices.intersection(selected):
        raise AssertionError("T6 repair repeats an open-T3 formation row")
    random.Random(f"{seed}:eedi-open-t3-t6-repair:{variant}").shuffle(selected)

    formal_ids = {str(row["example_id"]) for row in formal_rows}
    selected_ids = [str(train_rows[index]["example_id"]) for index in selected]
    if formal_ids.intersection(selected_ids):
        raise ValueError("T6 repair overlaps the formal evaluation set")
    task_counts = Counter(str(train_rows[index]["task"]) for index in selected)
    source_counts = Counter(str(train_rows[index]["source"]) for index in selected)
    label_counts = Counter(
        t6_label(train_rows[index])
        for index in selected
        if train_rows[index]["task"] == "predict_error_persistence"
    )
    return {
        "stage": "layerwise_task_retention_index_manifest_v1",
        # The trainer/evaluator already recognizes this branch and validates
        # its strict open-T3 prompt view. The curriculum metadata below
        # distinguishes this repair stage from the formation stage.
        "branch": "eedi_full",
        "seed": seed,
        "source_train_sha256": SOURCE_TRAIN_SHA256,
        "formal_eval_sha256": FORMAL_EVAL_SHA256,
        "formal_example_id_overlap": 0,
        "start_global_position": START_GLOBAL_POSITION,
        "segment_rows": SEGMENT_ROWS,
        "selected_example_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "t6_label_counts": dict(sorted(label_counts.items())),
        "indices": selected,
        "curriculum_mode": "open_t3_formation_then_t6_repair",
        "curriculum_stage": "t6_repair",
        "repair_variant": variant,
        "prior_manifest_sha256": PRIOR_MANIFEST_SHA256,
        "prior_segment_start": 12_000,
        "prior_segment_end": 16_999,
        "prior_segment_overlap": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if sha256(args.train_jsonl) != SOURCE_TRAIN_SHA256:
        raise ValueError("T6 repair source-train SHA256 drift")
    if sha256(args.formal_eval_jsonl) != FORMAL_EVAL_SHA256:
        raise ValueError("T6 repair formal-eval SHA256 drift")
    if sha256(args.prior_manifest) != PRIOR_MANIFEST_SHA256:
        raise ValueError("T6 repair prior-manifest SHA256 drift")
    train_rows = read_jsonl(args.train_jsonl)
    formal_rows = read_jsonl(args.formal_eval_jsonl)
    prior = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in SPECS:
        value = build(train_rows, formal_rows, prior, variant=variant, seed=args.seed)
        path = args.output_dir / f"layerwise_eedi_open_t3_t6_repair_{variant}_seed42_v1.json"
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        print(json.dumps({"path": str(path), "task_counts": value["task_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
