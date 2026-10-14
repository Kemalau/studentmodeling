#!/usr/bin/env python3
"""Freeze deterministic donor/recipient/control pairs for state-swap evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from state_swap_misconception_smoke import build_pairs, build_samemal_controls, read_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--task", default="malrule_next_response")
    parser.add_argument("--swap-target", choices=["identity", "response"], default="response")
    parser.add_argument("--num-pairs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl, args.task)
    pairs = build_pairs(rows, args.num_pairs, args.seed, args.swap_target)
    controls = build_samemal_controls(rows, pairs, args.seed + 1009)
    if len(pairs) != args.num_pairs:
        raise ValueError(f"Built {len(pairs)} pairs, expected {args.num_pairs}")
    if any(control is None for control in controls):
        raise ValueError("At least one frozen pair is missing its same-malrule control")

    payload = {
        "version": 1,
        "source_jsonl": str(args.jsonl),
        "source_sha256": sha256(args.jsonl),
        "task": args.task,
        "swap_target": args.swap_target,
        "seed": args.seed,
        "same_malrule_seed": args.seed + 1009,
        "num_pairs": len(pairs),
        "pairs": [
            {
                "donor_id": donor["example_id"],
                "recipient_id": recipient["example_id"],
                "samemal_id": control["example_id"],
            }
            for (donor, recipient), control in zip(pairs, controls, strict=True)
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
