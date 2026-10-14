#!/usr/bin/env python3
"""Build a fresh train-only panel for State8 collapse-repair ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import build_layerwise_task_retention_manifests as manifests
except ImportError:  # package-style unit tests
    import scripts.build_layerwise_task_retention_manifests as manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    train = manifests.read_jsonl(args.train_jsonl)
    formal = manifests.read_jsonl(args.formal_eval_jsonl)
    prior_indices: set[int] = set()
    prior_receipts: dict[str, str] = {}
    prior_sets: dict[str, set[int]] = {}
    for path in args.prior_manifest:
        value = json.loads(path.read_text(encoding="utf-8"))
        indices = {int(index) for index in value.get("indices") or []}
        if len(indices) != int(value.get("segment_rows", -1)):
            raise ValueError(f"Excluded manifest cardinality drift: {path}")
        prior_indices.update(indices)
        prior_sets[str(path)] = indices
        prior_receipts[str(path)] = manifests.sha256(path)

    branch = "stateonly_repair_fresh"
    manifests.BRANCH_SPECS[branch] = {
        "t3": {"eedi_mining": 700, "malrule": 100},
        "t6": {},
    }
    # Standard anchors contribute the remaining 100 examples, retaining a
    # small non-T3 panel while diagnosis dominates the repair signal.
    value = manifests.build(
        train,
        formal,
        branch=branch,
        seed=args.seed,
        # This is intentional replay: the primary curriculum already consumed
        # the full EEDI train pool.  Formal-test separation, not fictitious
        # novelty, is the scientific boundary for this consolidation pilot.
        preused=None,
        start_global_position=14_000,
    )
    value.update(
        stage="state8_stateonly_repair_index_manifest_v1",
        branch=branch,
        prior_manifests=prior_receipts,
        prior_index_union_size=len(prior_indices),
        prior_overlap_by_manifest={
            name: len(indices.intersection(value["indices"]))
            for name, indices in sorted(prior_sets.items())
        },
        prior_index_union_overlap=len(prior_indices.intersection(value["indices"])),
        replay_is_intentional=True,
        stateonly_teacher_kd=True,
    )
    if value["formal_example_id_overlap"] != 0:
        raise AssertionError("Repair panel overlaps the formal evaluation set")
    value["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value[key] for key in (
        "stage", "segment_rows", "task_counts", "source_counts",
        "prior_index_union_size", "prior_index_union_overlap",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
