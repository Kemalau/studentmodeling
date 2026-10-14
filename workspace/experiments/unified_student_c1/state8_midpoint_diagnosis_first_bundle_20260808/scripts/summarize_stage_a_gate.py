#!/usr/bin/env python3
"""Summarize the fixed Stage-A C4 gate outputs without changing the gate set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


TASK = "malrule_next_response"
CONDITIONS = ("noop", "samemal", "random", "swap")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_task_metrics(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    metrics = raw["model"]["metrics"][TASK]
    keep = ("n", "accuracy", "macro_f1", "nll", "ece")
    return {key: metrics[key] for key in keep if key in metrics}


def probe_summary(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    result: dict[str, Any] = {
        "n_labels": raw["n_labels"],
        "random_level": raw["random_level"],
        "positions": {},
    }
    for position, layer_metrics in raw["metrics"].items():
        cross = {
            str(layer): {
                "accuracy": values["cross"]["accuracy"],
                "macro_f1": values["cross"]["macro_f1"],
            }
            for layer, values in layer_metrics.items()
        }
        best_layer = max(cross, key=lambda layer: (cross[layer]["accuracy"], cross[layer]["macro_f1"]))
        result["positions"][position] = {
            "cross_by_layer": cross,
            "best_cross_layer": int(best_layer),
            "best_cross": cross[best_layer],
        }
    return result


def mean(records: list[dict[str, Any]], key: str) -> float:
    return fmean(float(record[key]) for record in records)


def clean_swap_summary(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in raw["records"]:
        grouped[int(record["pair_idx"])][record["condition"]] = record

    clean_pair_ids = sorted(
        pair_idx
        for pair_idx, conditions in grouped.items()
        if conditions.get("none", {}).get("category") == "stay_B"
    )
    if not clean_pair_ids:
        raise ValueError("The fixed gate produced no clean pairs (none=stay_B)")

    output: dict[str, Any] = {
        "all_pairs": raw["summary"],
        "clean_definition": "none.category == stay_B",
        "clean_n": len(clean_pair_ids),
        "clean": {},
    }
    for condition in CONDITIONS:
        records = [grouped[pair_idx][condition] for pair_idx in clean_pair_ids]
        bases = [grouped[pair_idx]["none"] for pair_idx in clean_pair_ids]
        counts = {
            category: sum(record["category"] == category for record in records)
            for category in ("flip_to_A", "stay_B", "other")
        }
        n = len(records)
        output["clean"][condition] = {
            "n": n,
            "category_counts": counts,
            "flip_to_A_rate": counts["flip_to_A"] / n,
            "stay_B_rate": counts["stay_B"] / n,
            "other_rate": counts["other"] / n,
            "changed_top1_rate": sum(r["pred"] != b["pred"] for r, b in zip(records, bases)) / n,
            "mean_delta_p_A": fmean(float(r["p_A"]) - float(b["p_A"]) for r, b in zip(records, bases)),
            "mean_delta_p_B": fmean(float(r["p_B"]) - float(b["p_B"]) for r, b in zip(records, bases)),
            "mean_delta_margin_A_minus_B": fmean(
                (float(r["score_A"]) - float(r["score_B"]))
                - (float(b["score_A"]) - float(b["score_B"]))
                for r, b in zip(records, bases)
            ),
            "mean_p_A": mean(records, "p_A"),
            "mean_p_B": mean(records, "p_B"),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    gate_dir = args.gate_dir
    summary = {
        "response_same_masked": compact_task_metrics(gate_dir / "response_same_masked.json"),
        "response_cross_masked": compact_task_metrics(gate_dir / "response_cross_masked.json"),
        "state_probe_masked": probe_summary(gate_dir / "state_probe_masked.json"),
        "state_swap_masked": clean_swap_summary(gate_dir / "state_swap_masked.json"),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
