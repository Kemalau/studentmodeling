#!/usr/bin/env python3
"""Summarize frozen C4 real-donor control and trajectory dose response."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def paired_bootstrap(values: list[float], seed: int, samples: int) -> list[float]:
    rng = random.Random(seed)
    draws = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def summarize(payload: dict, bootstrap_samples: int = 10_000) -> dict:
    paired: dict[int, dict[str, dict]] = defaultdict(dict)
    for record in payload["records"]:
        paired[int(record["pair_idx"])][str(record["condition"])] = record
    required = {
        "none", "noop", "swap", "random", "samemal", "random_real",
        "dose_0p25", "dose_0p5", "dose_0p75",
    }
    complete = [row for row in paired.values() if required <= row.keys()]
    clean = [row for row in complete if row["none"]["category"] == "stay_B"]
    if not clean:
        raise ValueError("No clean native-stay_B pairs")

    conditions = ["noop", "dose_0p25", "dose_0p5", "dose_0p75", "swap"]
    alpha_by_condition = {
        "noop": 0.0, "dose_0p25": 0.25, "dose_0p5": 0.5,
        "dose_0p75": 0.75, "swap": 1.0,
    }
    dose = []
    for condition in conditions:
        dose.append({
            "condition": condition,
            "alpha": alpha_by_condition[condition],
            "flip_to_A_rate": mean([
                float(row[condition]["category"] == "flip_to_A") for row in clean
            ]),
            "mean_margin_A_minus_B": mean([
                row[condition]["score_A"] - row[condition]["score_B"] for row in clean
            ]),
            "mean_delta_margin_vs_noop": mean([
                (row[condition]["score_A"] - row[condition]["score_B"])
                - (row["noop"]["score_A"] - row["noop"]["score_B"])
                for row in clean
            ]),
            "mean_p_A": mean([row[condition]["p_A"] for row in clean]),
        })

    directed_minus_real = [
        float(row["swap"]["category"] == "flip_to_A")
        - float(row["random_real"]["category"] == "flip_to_A")
        for row in clean
    ]
    directed_minus_noise = [
        float(row["swap"]["category"] == "flip_to_A")
        - float(row["random"]["category"] == "flip_to_A")
        for row in clean
    ]
    collision = [
        row["random_real"].get("real_random_target_on_recipient_query")
        == row["random_real"]["donor_target"]
        for row in clean
    ]
    margin_steps = [row["mean_margin_A_minus_B"] for row in dose]
    flip_steps = [row["flip_to_A_rate"] for row in dose]
    return {
        "status": "complete",
        "protocol": "frozen seed42 CF1 adapter; preregistered 150 pairs; native-stay_B clean subset",
        "n_manifest_pairs": int(payload["num_pairs"]),
        "n_complete_pairs": len(complete),
        "clean_n": len(clean),
        "real_random": {
            "selection": payload.get("real_random_selection"),
            "seed": payload.get("real_random_seed"),
            "flip_to_A_rate": mean([
                float(row["random_real"]["category"] == "flip_to_A") for row in clean
            ]),
            "directed_flip_to_A_rate": mean([
                float(row["swap"]["category"] == "flip_to_A") for row in clean
            ]),
            "directed_minus_real_random": mean(directed_minus_real),
            "directed_minus_real_random_paired_bootstrap_95ci": paired_bootstrap(
                directed_minus_real, 20_260_904, bootstrap_samples
            ),
            "random_rule_answer_equals_directed_answer_rate": mean([float(x) for x in collision]),
        },
        "gaussian_noise_reference": {
            "flip_to_A_rate": mean([
                float(row["random"]["category"] == "flip_to_A") for row in clean
            ]),
            "directed_minus_gaussian_noise": mean(directed_minus_noise),
            "directed_minus_gaussian_noise_paired_bootstrap_95ci": paired_bootstrap(
                directed_minus_noise, 42, bootstrap_samples
            ),
        },
        "dose_response": {
            "formula": payload.get("dose_formula"),
            "rows": dose,
            "flip_rate_nondecreasing": all(a <= b for a, b in zip(flip_steps, flip_steps[1:])),
            "mean_margin_strictly_increasing": all(a < b for a, b in zip(margin_steps, margin_steps[1:])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = summarize(payload, args.bootstrap_samples)
    result["lineage"] = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
