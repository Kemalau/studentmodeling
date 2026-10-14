#!/usr/bin/env python3
"""Compare two teacher caches with strict token/top-k and float16 tolerances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_records(root: Path) -> dict[str, dict]:
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    loaded: dict[str, dict] = {}
    for row in index:
        filename = str(row["file"])
        if filename not in loaded:
            loaded[filename] = torch.load(
                root / filename, map_location="cpu", weights_only=True
            )
        example_id = str(row["example_id"])
        result[example_id] = loaded[filename][example_id]
    return result


def compare(left: Path, right: Path, tolerance: float) -> dict:
    left_records = load_records(left)
    right_records = load_records(right)
    if set(left_records) != set(right_records):
        raise ValueError("Cache example sets differ")
    maximum = 0.0
    for example_id in sorted(left_records):
        a = left_records[example_id]
        b = right_records[example_id]
        if list(a["target_token_ids"]) != list(b["target_token_ids"]):
            raise ValueError(f"Target-token mismatch: {example_id}")
        if not torch.equal(a["teacher_topk_ids"], b["teacher_topk_ids"]):
            raise ValueError(f"Top-k ID mismatch: {example_id}")
        delta = (
            a["teacher_topk_logprobs"].float()
            - b["teacher_topk_logprobs"].float()
        ).abs()
        if delta.numel():
            maximum = max(maximum, float(delta.max().item()))
    if maximum > tolerance:
        raise ValueError(
            f"Teacher log-prob drift {maximum} exceeds tolerance {tolerance}"
        )
    return {
        "status": "equivalent",
        "rows": len(left_records),
        "topk_ids_exact": True,
        "target_tokens_exact": True,
        "max_abs_logprob_delta": maximum,
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1 / 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right, args.tolerance)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
