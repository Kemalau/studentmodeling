#!/usr/bin/env python3
"""Write a machine-readable M2 transition gate without hiding failed science."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unroll", type=int, required=True)
    parser.add_argument("--projection-mode", required=True)
    parser.add_argument("--max-accuracy-gap", type=float, default=0.05)
    parser.add_argument("--min-state-variance", type=float, default=1e-8)
    parser.add_argument("--max-pairwise-cosine", type=float, default=0.9999)
    parser.add_argument("--min-candidate-n", type=int, default=50)
    parser.add_argument("--min-direct-accuracy", type=float, default=0.60)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    checks = {
        "candidate_support": int(metrics.get("candidate_n", 0)) >= args.min_candidate_n,
        "direct_task_sufficiency": float(metrics.get("direct_accuracy", 0.0))
        >= args.min_direct_accuracy,
        "accuracy_gap": float(metrics.get("accuracy_gap_recurrent_minus_direct", -1.0))
        >= -args.max_accuracy_gap,
        "nonconstant_variance": float(metrics.get("state_variance", 0.0))
        >= args.min_state_variance,
        "nonconstant_cosine": float(metrics.get("mean_pairwise_cosine", 1.0))
        <= args.max_pairwise_cosine,
    }
    payload = {
        "stage": "M2_recurrent_transition",
        "unroll": args.unroll,
        "projection_mode": args.projection_mode,
        "scientific_gate_pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_accuracy_gap": args.max_accuracy_gap,
            "min_state_variance": args.min_state_variance,
            "max_pairwise_cosine": args.max_pairwise_cosine,
            "min_candidate_n": args.min_candidate_n,
            "min_direct_accuracy": args.min_direct_accuracy,
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
