#!/usr/bin/env python3
"""Compare matched original-only and CF1 checkpoint causal curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def condition_for(path: Path) -> str:
    joined = "/".join(path.parts).lower()
    if "original-visionpad" in joined:
        return "original"
    if "cf1-visionpad" in joined:
        return "cf1"
    raise ValueError(f"Could not infer condition from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    curves: dict[str, dict[int, dict[str, Any]]] = {}
    for path in sorted(args.input_root.rglob("curve_summary.json")):
        condition = condition_for(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if condition in curves:
            raise ValueError(f"Duplicate {condition} curve at {path}")
        curves[condition] = {int(row["step"]): row for row in payload["checkpoints"]}
    if set(curves) != {"original", "cf1"}:
        raise ValueError(f"Expected original and cf1 curves, got {sorted(curves)}")
    steps = sorted(set(curves["original"]) & set(curves["cf1"]))
    if steps != sorted(curves["original"]) or steps != sorted(curves["cf1"]):
        raise ValueError("Original and CF1 checkpoint steps do not match")

    rows = []
    for step in steps:
        original = curves["original"][step]
        cf1 = curves["cf1"][step]
        rows.append(
            {
                "step": step,
                "original": original,
                "cf1": cf1,
                "cf1_minus_original_D_specific": cf1["D_specific"] - original["D_specific"],
            }
        )

    lines = [
        "# Original-only versus CF1 checkpoint causal curves",
        "",
        "| step | original swap | original random | original D_specific | CF1 swap | CF1 random | CF1 D_specific | CF1-original |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        original, cf1 = row["original"], row["cf1"]
        lines.append(
            f"| {row['step']} | {original['swap_flip_to_A']:.3f} | {original['random_flip_to_A']:.3f} | "
            f"{original['D_specific']:.3f} | {cf1['swap_flip_to_A']:.3f} | {cf1['random_flip_to_A']:.3f} | "
            f"{cf1['D_specific']:.3f} | {row['cf1_minus_original_D_specific']:.3f} |"
        )
    lines.extend(["", "All rows use the same frozen 150-pair manifest; clean subsets are checkpoint-specific."])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"checkpoints": rows}, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"compared {len(rows)} matched checkpoints -> {args.output_md}")


if __name__ == "__main__":
    main()
