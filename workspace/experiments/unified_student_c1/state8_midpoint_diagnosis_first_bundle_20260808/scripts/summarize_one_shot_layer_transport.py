#!/usr/bin/env python3
"""Aggregate grouped one-shot residual-stream patches on one clean subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CONDITIONS = {"none", "noop", "random", "samemal", "swap"}


def summarize_payload(payload: dict) -> list[dict]:
    grouped: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for record in payload.get("records") or []:
        grouped[str(record["layer_config"])][int(record["pair_idx"])][str(record["condition"])] = record
    rows = []
    for group, pairs in sorted(grouped.items()):
        if not pairs:
            raise ValueError(f"Empty layer group: {group}")
        if any(set(conditions) != CONDITIONS for conditions in pairs.values()):
            raise ValueError(f"Incomplete conditions in layer group: {group}")
        clean = [index for index, conditions in pairs.items() if conditions["none"]["category"] == "stay_B"]
        if not clean:
            raise ValueError(f"No clean native=B pairs in layer group: {group}")
        metrics = {}
        native = [pairs[index]["none"] for index in clean]
        for condition in ("noop", "random", "samemal", "swap"):
            records = [pairs[index][condition] for index in clean]
            counts = Counter(record["category"] for record in records)
            metrics[condition] = {
                "n": len(records),
                "flip_to_A_rate": counts["flip_to_A"] / len(records),
                "changed_top1_rate": sum(
                    record["pred"] != base["pred"]
                    for record, base in zip(records, native, strict=True)
                ) / len(records),
                "mean_delta_margin_A_minus_B": sum(
                    (record["score_A"] - record["score_B"])
                    - (base["score_A"] - base["score_B"])
                    for record, base in zip(records, native, strict=True)
                ) / len(records),
            }
        rows.append({
            "layer_config": group,
            "layers": pairs[clean[0]]["swap"]["layers"],
            "clean_n": len(clean),
            "D_specific": metrics["swap"]["flip_to_A_rate"] - metrics["random"]["flip_to_A_rate"],
            "D_damage": metrics["random"]["flip_to_A_rate"] - metrics["noop"]["flip_to_A_rate"],
            "conditions": metrics,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(summarize_payload(json.loads(path.read_text(encoding="utf-8"))))
    names = [row["layer_config"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate layer configurations: {names}")
    selected = max(rows, key=lambda row: (row["D_specific"], -row["D_damage"]))
    payload = {
        "stage": "one_shot_layer_transport_C4_prefix_screen",
        "diagnostic_only": True,
        "frozen_manifest_prefix_n": 50,
        "rows": sorted(rows, key=lambda row: row["layer_config"]),
        "selected_layer_config": selected["layer_config"],
        "full_trajectory_required": selected["layer_config"] == "all",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
