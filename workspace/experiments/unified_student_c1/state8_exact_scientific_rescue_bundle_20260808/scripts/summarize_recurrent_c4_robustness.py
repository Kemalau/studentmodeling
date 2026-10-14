#!/usr/bin/env python3
"""Add paired continuous-effect and bootstrap evidence to frozen recurrent C4.

This audit is diagnostic only.  It never replaces or changes the preregistered
top-1 scientific gate in ``evaluate_recurrent_causal_gate.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Callable


CONDITIONS = ("native", "noop", "swap", "random", "samemal")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty sample")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float], *, draws: int, seed: int
) -> dict[str, float | int]:
    if not values:
        raise ValueError("Bootstrap sample is empty")
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    rng = random.Random(seed)
    n = len(values)
    means = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(draws)
    ]
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def paired_values(
    rows: list[dict], function: Callable[[dict[str, dict]], float]
) -> list[float]:
    return [float(function(row)) for row in rows]


def summarize(payload: dict, *, draws: int = 10_000, seed: int = 20260804) -> dict:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("C4 payload has no records")
    by_pair: dict[int, dict[str, dict]] = {}
    for record in records:
        pair_idx = int(record["pair_idx"])
        condition = str(record["condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected C4 condition: {condition}")
        conditions = by_pair.setdefault(pair_idx, {})
        if condition in conditions:
            raise ValueError(f"Duplicate C4 condition pair={pair_idx} condition={condition}")
        conditions[condition] = record
    incomplete = {
        pair_idx: sorted(set(CONDITIONS) - set(conditions))
        for pair_idx, conditions in by_pair.items()
        if set(conditions) != set(CONDITIONS)
    }
    if incomplete:
        raise ValueError(f"Incomplete C4 pairs: {dict(list(incomplete.items())[:3])}")
    clean = [
        by_pair[pair_idx]
        for pair_idx in sorted(by_pair)
        if by_pair[pair_idx]["native"]["category"] == "stay_B"
    ]
    if not clean:
        raise ValueError("C4 payload has no native-clean pairs")

    effects = {
        "D_specific_paired_flip": paired_values(
            clean,
            lambda row: (row["swap"]["category"] == "flip_to_A")
            - (row["random"]["category"] == "flip_to_A"),
        ),
        "swap_minus_random_margin": paired_values(
            clean,
            lambda row: row["swap"]["margin_A_minus_B"]
            - row["random"]["margin_A_minus_B"],
        ),
        "swap_minus_random_p_A": paired_values(
            clean, lambda row: row["swap"]["p_A"] - row["random"]["p_A"]
        ),
        "swap_minus_native_margin": paired_values(
            clean,
            lambda row: row["swap"]["margin_A_minus_B"]
            - row["native"]["margin_A_minus_B"],
        ),
        "samemal_minus_native_margin": paired_values(
            clean,
            lambda row: row["samemal"]["margin_A_minus_B"]
            - row["native"]["margin_A_minus_B"],
        ),
    }
    bootstrap = {
        name: bootstrap_mean_ci(values, draws=draws, seed=seed + index)
        for index, (name, values) in enumerate(effects.items())
    }
    boundary_thresholds = (0.01, 0.05, 0.10, 0.25)
    boundary = {
        condition: {
            f"abs_margin_le_{threshold:g}": sum(
                abs(float(row[condition]["margin_A_minus_B"])) <= threshold
                for row in clean
            )
            for threshold in boundary_thresholds
        }
        for condition in CONDITIONS
    }
    recorded = payload.get("summary") or {}
    observed_d = bootstrap["D_specific_paired_flip"]["mean"]
    if abs(float(recorded.get("D_specific", observed_d)) - float(observed_d)) > 1e-12:
        raise ValueError("Paired D_specific does not match the frozen C4 summary")
    return {
        "stage": "recurrent_c4_robustness_diagnostic_v1",
        "diagnostic_only": True,
        "does_not_replace_scientific_gate": True,
        "all_pairs_n": len(by_pair),
        "clean_n": len(clean),
        "checkpoint": {
            "adapter": payload.get("adapter"),
            "recurrent_state": payload.get("recurrent_state"),
            "pair_manifest": payload.get("pair_manifest"),
        },
        "frozen_scientific_gate": {
            "D_specific": recorded.get("D_specific"),
            "checks": recorded.get("checks"),
            "scientific_gate_pass": recorded.get("scientific_gate_pass"),
        },
        "paired_bootstrap": bootstrap,
        "decision_boundary_counts_on_clean_pairs": boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.input.read_text(encoding="utf-8")),
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
