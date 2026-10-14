#!/usr/bin/env python3
"""Compare matched control and paired-state-swap C4 gate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def find_summary(root: Path, job_fragment: str) -> Path:
    matches = [path for path in root.rglob("gate_summary.json") if job_fragment in str(path)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {job_fragment} gate summary, found {matches}")
    return matches[0]


def compact(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    swap = raw["state_swap_masked"]
    clean = swap["clean"]
    return {
        "source": str(path),
        "same_accuracy": raw["response_same_masked"]["accuracy"],
        "cross_accuracy": raw["response_cross_masked"]["accuracy"],
        "probe_cross_accuracy": raw["state_probe_masked"]["positions"]["state"]["best_cross"]["accuracy"],
        "probe_layer": raw["state_probe_masked"]["positions"]["state"]["best_cross_layer"],
        "clean_n": swap["clean_n"],
        "swap_flip": clean["swap"]["flip_to_A_rate"],
        "random_flip": clean["random"]["flip_to_A_rate"],
        "same_malrule_flip": clean["samemal"]["flip_to_A_rate"],
        "noop_flip": clean["noop"]["flip_to_A_rate"],
        "causal_specificity": clean["swap"]["flip_to_A_rate"] - clean["random"]["flip_to_A_rate"],
        "random_damage": clean["random"]["flip_to_A_rate"] - clean["noop"]["flip_to_A_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    control = compact(find_summary(args.input_root, "paired-state-control-seed42"))
    treatment = compact(find_summary(args.input_root, "paired-state-swap-seed42"))
    result = {
        "control": control,
        "paired_swap": treatment,
        "delta": {
            key: treatment[key] - control[key]
            for key in ("same_accuracy", "cross_accuracy", "probe_cross_accuracy", "causal_specificity")
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [
        "# Paired state-swap A/B comparison",
        "",
        "| arm | same acc | cross acc | probe acc | clean n | swap | random | same-mal | noop | D_specific | D_damage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in (("control", control), ("paired-swap", treatment)):
        rows.append(
            f"| {name} | {values['same_accuracy']:.3f} | {values['cross_accuracy']:.3f} | "
            f"{values['probe_cross_accuracy']:.3f} | {values['clean_n']} | {values['swap_flip']:.3f} | "
            f"{values['random_flip']:.3f} | {values['same_malrule_flip']:.3f} | {values['noop_flip']:.3f} | "
            f"{values['causal_specificity']:.3f} | {values['random_damage']:.3f} |"
        )
    rows.extend(
        [
            "",
            f"Treatment-control delta: cross accuracy {result['delta']['cross_accuracy']:+.3f}; "
            f"causal specificity {result['delta']['causal_specificity']:+.3f}.",
        ]
    )
    args.output_md.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
