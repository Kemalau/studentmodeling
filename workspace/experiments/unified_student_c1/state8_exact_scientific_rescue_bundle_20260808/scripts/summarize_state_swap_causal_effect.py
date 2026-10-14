#!/usr/bin/env python3
"""Summarize donor-state swap effects relative to paired controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(payload: dict) -> dict:
    paired = defaultdict(dict)
    for record in payload["records"]:
        paired[(record["layer_config"], int(record["pair_idx"]))][
            record["condition"]
        ] = record
    output = {}
    for layer_config in sorted({key[0] for key in paired}):
        rows = [conditions for (name, _), conditions in paired.items() if name == layer_config]
        complete = [row for row in rows if {"none", "noop", "random", "samemal", "swap"} <= row.keys()]
        clean = [row for row in complete if row["none"]["category"] == "stay_B"]
        output[layer_config] = {
            "n_complete_pairs": len(complete),
            "n_clean_none_stay_B": len(clean),
            "unconditional": {
                "swap_flip_to_A_rate": sum(row["swap"]["category"] == "flip_to_A" for row in complete)
                / max(1, len(complete)),
                "noop_flip_to_A_rate": sum(row["noop"]["category"] == "flip_to_A" for row in complete)
                / max(1, len(complete)),
                "random_flip_to_A_rate": sum(row["random"]["category"] == "flip_to_A" for row in complete)
                / max(1, len(complete)),
            },
            "clean_none_stay_B": {
                "swap_flip_to_A_rate": sum(row["swap"]["category"] == "flip_to_A" for row in clean)
                / max(1, len(clean)),
                "noop_flip_to_A_rate": sum(row["noop"]["category"] == "flip_to_A" for row in clean)
                / max(1, len(clean)),
                "random_flip_to_A_rate": sum(row["random"]["category"] == "flip_to_A" for row in clean)
                / max(1, len(clean)),
                "samemal_flip_to_A_rate": sum(row["samemal"]["category"] == "flip_to_A" for row in clean)
                / max(1, len(clean)),
                "swap_minus_noop_flip_rate": (
                    sum(row["swap"]["category"] == "flip_to_A" for row in clean)
                    - sum(row["noop"]["category"] == "flip_to_A" for row in clean)
                )
                / max(1, len(clean)),
                "swap_minus_random_flip_rate": (
                    sum(row["swap"]["category"] == "flip_to_A" for row in clean)
                    - sum(row["random"]["category"] == "flip_to_A" for row in clean)
                )
                / max(1, len(clean)),
                "mean_delta_p_A_swap_minus_noop": sum(
                    row["swap"]["p_A"] - row["noop"]["p_A"] for row in clean
                )
                / max(1, len(clean)),
                "mean_delta_margin_swap_minus_noop": sum(
                    (row["swap"]["score_A"] - row["swap"]["score_B"])
                    - (row["noop"]["score_A"] - row["noop"]["score_B"])
                    for row in clean
                )
                / max(1, len(clean)),
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
