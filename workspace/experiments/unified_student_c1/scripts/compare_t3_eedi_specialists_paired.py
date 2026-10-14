#!/usr/bin/env python3
"""Paired Eedi-only T3 semantic comparisons on the frozen 894-row panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


EXPECTED = 894


def read(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [
        row for row in rows
        if row.get("task") == "diagnose_misconception" and row.get("source") == "eedi_mining"
    ]
    indexed = {str(row["example_id"]): row for row in rows}
    if len(rows) != EXPECTED or len(indexed) != EXPECTED:
        raise ValueError(f"Eedi support/duplicate drift rows={len(rows)} ids={len(indexed)}")
    if any("llm_judge_correct" not in row for row in rows):
        raise ValueError("Every Eedi row must have an independent semantic judgment")
    return indexed


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    observed = math.comb(trials, successes) / (2**trials)
    return min(1.0, sum(
        math.comb(trials, value) / (2**trials)
        for value in range(trials + 1)
        if math.comb(trials, value) / (2**trials) <= observed + 1e-15
    ))


def compare(
    challenger: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if set(challenger) != set(reference):
        raise ValueError("Eedi example-ID sets differ")
    ids = sorted(challenger)
    c = [bool(challenger[key]["llm_judge_correct"]) for key in ids]
    r = [bool(reference[key]["llm_judge_correct"]) for key in ids]
    challenger_only = sum(cv and not rv for cv, rv in zip(c, r, strict=True))
    reference_only = sum(rv and not cv for cv, rv in zip(c, r, strict=True))
    both_correct = sum(cv and rv for cv, rv in zip(c, r, strict=True))
    both_wrong = EXPECTED - challenger_only - reference_only - both_correct
    differences = [float(cv) - float(rv) for cv, rv in zip(c, r, strict=True)]
    rng = random.Random(seed)
    boot = [
        sum(differences[rng.randrange(EXPECTED)] for _ in range(EXPECTED)) / EXPECTED
        for _ in range(bootstrap_samples)
    ]
    return {
        "n": EXPECTED,
        "challenger_primary": sum(c) / EXPECTED,
        "reference_primary": sum(r) / EXPECTED,
        "challenger_minus_reference": sum(differences) / EXPECTED,
        "paired_bootstrap_95ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "discordant": {
            "challenger_only_correct": challenger_only,
            "reference_only_correct": reference_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "mcnemar_exact_two_sided_p": binomial_two_sided(
            challenger_only, challenger_only + reference_only
        ),
        "example_id_set_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--challenger-name", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    value = compare(
        read(args.challenger), read(args.reference),
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    value.update({
        "stage": "t3_eedi_specialist_paired_v1",
        "challenger": args.challenger_name,
        "reference": args.reference_name,
        "judge_role": "independent_trapi_v2_semantic",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
