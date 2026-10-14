#!/usr/bin/env python3
"""Select a full-fit candidate from authoritative semantic T3 and C4 screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": "eedi",
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": "cross",
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": "same",
}
VARIANTS = {
    "c4_g005": ("semantic_c4_g005", "state8-semantic-c4-g005-unified-finalfit-seed42"),
    "c4_g010": ("semantic_c4_g010", "state8-semantic-c4-g010-unified-finalfit-seed42"),
    "c4_g025": ("semantic_c4_g025", "state8-semantic-c4-g025-unified-finalfit-seed42"),
    "eedi_g005": ("semantic_eedi_g005", "state8-semantic-eedi-g005-unified-finalfit-seed42"),
    "eedi_g010": ("semantic_eedi_g010", "state8-semantic-eedi-g010-unified-finalfit-seed42"),
    "eedi_g025": ("semantic_eedi_g025", "state8-semantic-eedi-g025-unified-finalfit-seed42"),
    "eedi_g050": ("semantic_eedi_g050", "state8-semantic-eedi-g050-unified-finalfit-seed42"),
    "eedi_g025_single": (
        "semantic_eedi_g025_single_seed46",
        "state8-semantic-eedi-g025-single-seed46-unified-finalfit-seed42",
    ),
}
SELECTION_ELIGIBLE = frozenset(
    {"c4_g005", "c4_g010", "c4_g025", "eedi_g005", "eedi_g010", "eedi_g025", "eedi_g050"}
)
MIN_EEDI_SEMANTIC_ACCURACY = 0.17
MIN_SYNTHETIC_SEMANTIC_ACCURACY = 0.45
MIN_SEMANTIC_D_SPECIFIC = 0.10
MIN_SWAP_TO_DONOR = 0.40
MAX_RANDOM_MATCHES_EITHER = 0.30


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def score(slug: str, t3: dict[str, Any], c4: dict[str, Any]) -> dict[str, Any]:
    if slug not in VARIANTS:
        raise ValueError(f"Unknown candidate slug: {slug}")
    if t3.get("judge_backend") != "trapi" or t3.get("judge_errors") != 0:
        raise ValueError(f"Incomplete authoritative T3: {slug}")
    groups = t3.get("metrics_by_group") or {}
    if set(groups) != set(GROUPS):
        raise ValueError(f"T3 group drift: {slug}")
    values: dict[str, float] = {}
    schema_numerator = 0.0
    support = 0
    for name, short in GROUPS.items():
        row = groups[name]
        if int(row.get("n", -1)) != 100 or int(row.get("llm_judge_n", -1)) != 100:
            raise ValueError(f"T3 support drift: {slug}/{short}")
        if int(row.get("llm_judge_api_or_cache", -1)) != 100:
            raise ValueError(
                f"T3 group lacks independent semantic judgments: {slug}/{short}"
            )
        if int(row.get("llm_judge_exact_shortcuts", -1)) != 0:
            raise ValueError(f"Exact shortcut is not authoritative: {slug}/{short}")
        values[short] = float(row["llm_judge_accuracy"])
        schema_numerator += float(row["natural_schema_valid_rate"]) * 100
        support += 100
    if c4.get("semantic_judge_authoritative") is not True:
        raise ValueError(f"C4 semantic judge missing: {slug}")
    if int(c4.get("generation", {}).get("num_pairs", -1)) != 50:
        raise ValueError(f"C4 support drift: {slug}")
    synthetic = 0.5 * (values["cross"] + values["same"])
    d_specific = float(c4["semantic_D_specific"])
    swap = float(c4["cells"]["swap_to_donor"]["semantic_equivalence_rate"])
    noop = float(c4["noop_recipient_retention"])
    same_rule = float(c4["same_rule_recipient_retention"])
    random_any = float(c4["random_matches_either_stable_target_rate"])
    schema = schema_numerator / support
    qualified = (
        values["eedi"] >= MIN_EEDI_SEMANTIC_ACCURACY
        and synthetic >= MIN_SYNTHETIC_SEMANTIC_ACCURACY
        and schema >= 0.95
        and d_specific >= MIN_SEMANTIC_D_SPECIFIC
        and swap >= MIN_SWAP_TO_DONOR
        and random_any <= MAX_RANDOM_MATCHES_EITHER
    )
    aggregate = (
        0.30 * values["eedi"]
        + 0.25 * synthetic
        + 0.20 * max(0.0, d_specific)
        + 0.10 * swap
        + 0.05 * noop
        + 0.05 * same_rule
        + 0.05 * schema
        - 0.10 * random_any
    )
    variant, job = VARIANTS[slug]
    return {
        "slug": slug,
        "finalfit_variant": variant,
        "finalfit_job": job,
        "qualified": qualified,
        "selection_eligible": slug in SELECTION_ELIGIBLE,
        "score": aggregate,
        "checks": {
            "eedi_semantic_accuracy": values["eedi"],
            "synthetic_semantic_accuracy": synthetic,
            "natural_schema_valid_rate": schema,
            "semantic_D_specific": d_specific,
            "swap_to_donor": swap,
            "noop_recipient_retention": noop,
            "same_rule_recipient_retention": same_rule,
            "random_matches_either": random_any,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", nargs=3, action="append", metavar=("SLUG", "T3", "C4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [score(slug, read(Path(t3)), read(Path(c4))) for slug, t3, c4 in args.candidate]
    qualified = [
        row for row in rows if row["qualified"] and row["selection_eligible"]
    ]
    selected = max(qualified, key=lambda row: row["score"]) if qualified else None
    result = {
        "stage": "state8_semantic_candidate_selection_v1",
        "pass": selected is not None,
        "selected": selected,
        "candidates": sorted(rows, key=lambda row: row["slug"]),
        "selection_rule": (
            "T3 EEDI>=0.17, synthetic>=0.45, schema>=0.95; "
            "C4 D_specific>=0.10, swap>=0.40, random<=0.30; then weighted score"
        ),
        "screen_scope": "promotion_only_full1494_and_multiseed_final_gate_required",
        "seed46_role": "replication_only_never_hyperparameter_selection",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if selected else 2)


if __name__ == "__main__":
    main()
