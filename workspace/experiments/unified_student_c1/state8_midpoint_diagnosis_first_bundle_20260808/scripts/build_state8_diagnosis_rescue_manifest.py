#!/usr/bin/env python3
"""Build a disjoint 1k-row State8 curriculum after the frozen v5 segment."""

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
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--start-global-position", type=int, default=13000)
    args = parser.parse_args()
    train = manifests.read_jsonl(args.train_jsonl)
    formal = manifests.read_jsonl(args.formal_eval_jsonl)
    prior = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
    prior_indices = {int(value) for value in prior.get("indices") or []}
    if len(prior_indices) != int(prior.get("segment_rows", -1)):
        raise ValueError("Prior State8 manifest cardinality drift")
    # The frozen v5 segment consumes 10/15 available DBE transient T6 rows.
    # Keep the same 700/100/200 T3/T6/anchor task mix without replacement by
    # drawing the balanced T6 support from fresh FoundationalAssist examples.
    branch = "state8_diagnosis_causal_rescue"
    manifests.BRANCH_SPECS[branch] = {
        "t3": {"eedi_mining": 600, "malrule": 100},
        "t6": {
            ("foundationalassist", "persistent"): 50,
            ("foundationalassist", "transient"): 50,
        },
    }
    value = manifests.build(
        train,
        formal,
        branch=branch,
        seed=args.seed,
        preused=prior_indices,
        start_global_position=args.start_global_position,
    )
    value.update(
        stage="layerwise_task_retention_index_manifest_v1",
        experiment_stage="state8_diagnosis_causal_rescue_index_manifest_v1",
        branch="state8_diagnosis_causal_rescue",
        prior_manifest_sha256=manifests.sha256(args.prior_manifest),
        prior_segment_overlap=len(prior_indices.intersection(value["indices"])),
        diagnosis_causal_supervision=True,
    )
    if value["prior_segment_overlap"] != 0:
        raise AssertionError("Diagnosis rescue overlaps the frozen v5 segment")
    value["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value[key] for key in (
        "stage", "branch", "start_global_position", "segment_rows",
        "prior_segment_overlap", "task_counts", "source_counts",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
