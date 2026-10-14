#!/usr/bin/env python3
"""Compare control, all-layer, and late-half quartet-v2 causal gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "quartet-control-seed42": "control",
    "quartet-swap-all-seed42": "swap_all",
    "quartet-swap-latehalf-seed42": "swap_latehalf",
}


def compact(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    clean = raw["state_swap_masked"]["clean"]
    result = {
        "source": str(path),
        "same_accuracy": raw["response_same_masked"]["accuracy"],
        "cross_accuracy": raw["response_cross_masked"]["accuracy"],
        "probe_cross_accuracy": raw["state_probe_masked"]["positions"]["state"][
            "best_cross"
        ]["accuracy"],
        "clean_n": raw["state_swap_masked"]["clean_n"],
        "swap_flip": clean["swap"]["flip_to_A_rate"],
        "random_flip": clean["random"]["flip_to_A_rate"],
        "samemal_flip": clean["samemal"]["flip_to_A_rate"],
        "noop_flip": clean["noop"]["flip_to_A_rate"],
    }
    result["causal_specificity"] = result["swap_flip"] - result["random_flip"]
    result["random_damage"] = result["random_flip"] - result["noop_flip"]
    result["gate_pass"] = (
        result["same_accuracy"] >= 0.80
        and result["cross_accuracy"] >= 0.80
        and result["probe_cross_accuracy"] >= 0.85
        and result["causal_specificity"] >= 0.15
        and result["noop_flip"] <= 0.05
        and result["samemal_flip"] <= 0.10
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for fragment, name in EXPECTED.items():
        matches = [
            path
            for path in args.input_root.rglob("gate_summary.json")
            if fragment in str(path)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one summary for {fragment}, found {matches}")
        rows[name] = compact(matches[0])
    passing = [name for name in ("swap_all", "swap_latehalf") if rows[name]["gate_pass"]]
    selected = (
        max(
            passing,
            key=lambda name: (
                rows[name]["causal_specificity"],
                rows[name]["cross_accuracy"],
            ),
        )
        if passing
        else None
    )
    payload = {
        "stage": "quartet_v2_causal_identification",
        "scientific_gate_pass": selected is not None,
        "selected_condition": selected,
        "selection_rule": "pass all frozen gates; maximize D_specific then cross accuracy",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Quartet-v2 causal comparison",
        "",
        "| condition | pass | same | cross | probe | n | swap | random | same-mal | noop | D_specific | D_damage |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['gate_pass']} | {row['same_accuracy']:.3f} | "
            f"{row['cross_accuracy']:.3f} | {row['probe_cross_accuracy']:.3f} | "
            f"{row['clean_n']} | {row['swap_flip']:.3f} | {row['random_flip']:.3f} | "
            f"{row['samemal_flip']:.3f} | {row['noop_flip']:.3f} | "
            f"{row['causal_specificity']:.3f} | {row['random_damage']:.3f} |"
        )
    lines.extend(["", f"Selected condition: `{selected}`", ""])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
