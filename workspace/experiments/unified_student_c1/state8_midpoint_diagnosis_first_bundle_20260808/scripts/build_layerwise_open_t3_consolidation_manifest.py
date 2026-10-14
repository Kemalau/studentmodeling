#!/usr/bin/env python3
"""Build a disjoint phase-2 curriculum after open-T3 formation.

The first equal-T3/T6 segment occupies global positions 12000:13000.  This
manifest preserves its task/source proportions while selecting a fresh set of
train-only rows for positions 13000:14000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_layerwise_task_retention_manifests import (
    FORMAL_EVAL_SHA256,
    SOURCE_TRAIN_SHA256,
    build,
    read_jsonl,
    sha256,
)


PRIOR_MANIFEST_SHA256 = (
    "32adfcc7bd1d78f541ec8685ec8f96f9934b35471011bb896c402b1e77a7bf6f"
)
START_GLOBAL_POSITION = 13_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if sha256(args.train_jsonl) != SOURCE_TRAIN_SHA256:
        raise ValueError("Consolidation source-train SHA256 drift")
    if sha256(args.formal_eval_jsonl) != FORMAL_EVAL_SHA256:
        raise ValueError("Consolidation formal-eval SHA256 drift")
    if sha256(args.prior_manifest) != PRIOR_MANIFEST_SHA256:
        raise ValueError("Consolidation prior-manifest SHA256 drift")

    train_rows = read_jsonl(args.train_jsonl)
    formal_rows = read_jsonl(args.formal_eval_jsonl)
    prior = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
    prior_indices = {int(index) for index in prior["indices"]}
    if len(prior_indices) != 1_000:
        raise ValueError("Prior segment support drift")

    value = build(
        train_rows,
        formal_rows,
        branch="equal_t3_t6_consolidation",
        seed=args.seed,
        preused=prior_indices,
        start_global_position=START_GLOBAL_POSITION,
    )
    value.update(
        stage="layerwise_task_retention_index_manifest_v1",
        branch="equal_t3_t6",
        curriculum_mode="open_t3_formation_then_causal_task_consolidation",
        curriculum_stage="causal_task_consolidation",
        prior_manifest_sha256=PRIOR_MANIFEST_SHA256,
        prior_segment_start=12_000,
        prior_segment_end=12_999,
        source_shift_reason=(
            "phase-1 exhausts DBE transient T6 support; phase-2 preserves "
            "global T6 label balance with fresh FoundationalAssist rows"
        ),
    )
    selected = {int(index) for index in value["indices"]}
    if selected.intersection(prior_indices):
        raise AssertionError("Consolidation manifest repeats phase-1 rows")
    selected_ids = [str(train_rows[index]["example_id"]) for index in value["indices"]]
    value["selected_example_ids_sha256"] = hashlib.sha256(
        "\n".join(selected_ids).encode("utf-8")
    ).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "start_global_position": value["start_global_position"],
                "segment_rows": value["segment_rows"],
                "prior_segment_overlap": value["prior_segment_overlap"],
                "task_counts": value["task_counts"],
                "source_counts": value["source_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
