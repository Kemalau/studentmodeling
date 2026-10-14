#!/usr/bin/env python3
"""Summarize fixed-manifest C4 swap evaluations across training checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from summarize_stage_a_gate import clean_swap_summary


STEP_RE = re.compile(r"(?:checkpoint-|step_)(\d+)")


def checkpoint_step(path: Path) -> int:
    for part in reversed(path.parts):
        match = STEP_RE.search(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer checkpoint step from {path}")


def row_for(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    summary = clean_swap_summary(path)
    clean = summary["clean"]
    swap = clean["swap"]["flip_to_A_rate"]
    random = clean["random"]["flip_to_A_rate"]
    noop = clean["noop"]["flip_to_A_rate"]
    samemal = clean["samemal"]["flip_to_A_rate"]
    return {
        "step": checkpoint_step(path),
        "clean_n": summary["clean_n"],
        "swap_flip_to_A": swap,
        "random_flip_to_A": random,
        "samemal_flip_to_A": samemal,
        "noop_flip_to_A": noop,
        "D_specific": swap - random,
        "D_damage": random - noop,
        "swap_changed_top1": clean["swap"]["changed_top1_rate"],
        "swap_delta_margin": clean["swap"]["mean_delta_margin_A_minus_B"],
        "adapter": raw.get("adapter"),
        "pair_manifest": raw.get("pair_manifest"),
        "source": str(path),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Checkpoint causal curve",
        "",
        "| step | clean n | swap | random | same-rule | no-op | D_specific | D_damage | changed top-1 | delta margin |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {step} | {clean_n} | {swap} | {random} | {samemal} | {noop} | {specific} | {damage} | {changed} | {margin} |".format(
                step=row["step"],
                clean_n=row["clean_n"],
                swap=fmt(row["swap_flip_to_A"]),
                random=fmt(row["random_flip_to_A"]),
                samemal=fmt(row["samemal_flip_to_A"]),
                noop=fmt(row["noop_flip_to_A"]),
                specific=fmt(row["D_specific"]),
                damage=fmt(row["D_damage"]),
                changed=fmt(row["swap_changed_top1"]),
                margin=fmt(row["swap_delta_margin"]),
            )
        )
    lines.extend(
        [
            "",
            "`D_specific = swap - random`; `D_damage = random - no-op`. Clean pairs are defined independently at each checkpoint by `none.category == stay_B`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--filename", default="state_swap_masked.json")
    args = parser.parse_args()

    paths = sorted(args.input_root.rglob(args.filename))
    if not paths:
        raise SystemExit(f"No {args.filename} files found under {args.input_root}")
    rows = sorted((row_for(path) for path in paths), key=lambda row: row["step"])
    steps = [row["step"] for row in rows]
    if len(steps) != len(set(steps)):
        raise ValueError(f"Duplicate checkpoint steps: {steps}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"checkpoints": rows}, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(markdown(rows), encoding="utf-8")
    print(f"summarized {len(rows)} checkpoints -> {args.output_md}")


if __name__ == "__main__":
    main()
