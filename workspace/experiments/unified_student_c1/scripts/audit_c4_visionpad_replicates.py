#!/usr/bin/env python3
"""Recompute the frozen CF1 + vision-pad C4 replication audit from raw records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = ("swap", "random", "samemal", "noop")
PAIR_FIELDS = (
    "donor_id",
    "recipient_id",
    "donor_label",
    "recipient_label",
    "donor_target",
    "recipient_target",
    "samemal_id",
)


def load_records(path: Path) -> dict[int, dict[str, dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for record in value["records"]:
        pairs.setdefault(int(record["pair_idx"]), {})[str(record["condition"])] = record
    required = {"none", *CONDITIONS}
    incomplete = [index for index, arms in pairs.items() if set(arms) != required]
    if incomplete:
        raise ValueError(f"Incomplete condition sets in {path}: {incomplete[:5]}")
    return pairs


def pair_signature(pair: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(pair["none"].get(field) for field in PAIR_FIELDS)


def flip_indicator(pair: dict[str, dict[str, Any]], condition: str) -> int:
    return int(pair[condition]["category"] == "flip_to_A")


def margin_shift(pair: dict[str, dict[str, Any]], condition: str) -> float:
    def margin(record: dict[str, Any]) -> float:
        if "margin_A_minus_B" in record:
            return float(record["margin_A_minus_B"])
        return float(record["score_A"]) - float(record["score_B"])

    return margin(pair[condition]) - margin(pair["none"])


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    pairs: dict[int, dict[str, dict[str, Any]]],
    pair_ids: list[int],
    seed: int,
    replicates: int,
) -> dict[str, float]:
    effects = [
        flip_indicator(pairs[index], "swap")
        - flip_indicator(pairs[index], "random")
        for index in pair_ids
    ]
    generator = random.Random(seed)
    draws = [
        statistics.fmean(generator.choice(effects) for _ in effects)
        for _ in range(replicates)
    ]
    return {
        "mean": statistics.fmean(effects),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "replicates": replicates,
    }


def summarize(
    pairs: dict[int, dict[str, dict[str, Any]]], pair_ids: list[int]
) -> dict[str, Any]:
    rates = {
        condition: statistics.fmean(
            flip_indicator(pairs[index], condition) for index in pair_ids
        )
        for condition in CONDITIONS
    }
    shifts = {
        condition: statistics.fmean(
            margin_shift(pairs[index], condition) for index in pair_ids
        )
        for condition in CONDITIONS
    }
    return {
        "n": len(pair_ids),
        "flip_to_A_rate": rates,
        "D_specific": rates["swap"] - rates["random"],
        "mean_margin_shift": shifts,
        "continuous_specificity": shifts["swap"] - shifts["random"],
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed13", type=Path, required=True)
    parser.add_argument("--seed42", type=Path, required=True)
    parser.add_argument("--seed97", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    args = parser.parse_args()

    paths = {13: args.seed13, 42: args.seed42, 97: args.seed97}
    by_seed = {seed: load_records(path) for seed, path in paths.items()}
    reference_ids = sorted(by_seed[13])
    if any(sorted(pairs) != reference_ids for pairs in by_seed.values()):
        raise ValueError("The replicate files have different pair indices")
    pair_manifest_equal = all(
        pair_signature(by_seed[seed][index]) == pair_signature(by_seed[13][index])
        for seed in (42, 97)
        for index in reference_ids
    )
    if not pair_manifest_equal:
        raise ValueError("The replicate files do not contain identical frozen pairs")

    own_clean = {
        seed: [
            index
            for index in reference_ids
            if pairs[index]["none"]["category"] == "stay_B"
        ]
        for seed, pairs in by_seed.items()
    }
    common_clean = sorted(set.intersection(*(set(ids) for ids in own_clean.values())))
    result = {
        "pair_manifest_equal": pair_manifest_equal,
        "pair_count": len(reference_ids),
        "common_clean_n": len(common_clean),
        "input_sha256": {str(seed): sha256(path) for seed, path in paths.items()},
        "seeds": {},
    }
    for seed, pairs in by_seed.items():
        result["seeds"][str(seed)] = {
            "own_clean": summarize(pairs, own_clean[seed]),
            "common_clean": summarize(pairs, common_clean),
            "common_clean_paired_bootstrap": paired_bootstrap(
                pairs,
                common_clean,
                seed=20260823 + seed,
                replicates=args.bootstrap_replicates,
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
