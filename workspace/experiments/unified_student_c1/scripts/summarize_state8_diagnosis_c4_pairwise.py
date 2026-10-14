#!/usr/bin/env python3
"""Summarize pairwise output invariance alongside gold-referenced C4 metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from t3_judge_utils import read_jsonl
except ImportError:  # package-style tests
    from scripts.t3_judge_utils import read_jsonl


CONDITIONS = ("noop", "samemal", "random", "swap")


def summarize(rows: list[dict[str, Any]], panel: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("condition") or "")
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected pairwise C4 condition: {condition}")
        if row.get("reference_role") != "native_output":
            raise ValueError("Pairwise C4 reference-role drift")
        if "llm_judge_semantic_equivalent" not in row:
            raise ValueError("Pairwise C4 row lacks an authoritative semantic judgment")
        grouped[condition].append(row)
    if set(grouped) != set(CONDITIONS):
        raise ValueError(f"Pairwise C4 condition matrix drift: {sorted(grouped)}")
    eligible = int(panel.get("eligible_native_pairs", -1))
    if eligible <= 0 or any(len(grouped[name]) != eligible for name in CONDITIONS):
        raise ValueError("Pairwise C4 support drift")

    rates = {
        condition: sum(
            bool(row["llm_judge_semantic_equivalent"])
            for row in grouped[condition]
        )
        / eligible
        for condition in CONDITIONS
    }
    return {
        "stage": "state8_diagnosis_c4_pairwise_summary_v1",
        "status": "complete",
        "semantic_judge_authoritative": True,
        "schema_authoritative": False,
        "eligible_native_pairs": eligible,
        "excluded_native_pairs": list(panel.get("excluded_native_pairs") or []),
        "noop_native_output_invariance": rates["noop"],
        "same_rule_native_output_invariance": rates["samemal"],
        "random_native_output_invariance": rates["random"],
        "swap_native_output_invariance": rates["swap"],
        "same_rule_vs_swap_separation": rates["samemal"] - rates["swap"],
        "condition_rates": rates,
        "interpretation": (
            "Pairwise diagnosis invariance is distinct from gold task correctness; "
            "the latter remains authoritative in the ordinary semantic C4 panel."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-redacted", type=Path, required=True)
    parser.add_argument("--panel-receipt", type=Path, required=True)
    parser.add_argument("--gold-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        read_jsonl(args.judge_redacted),
        json.loads(args.panel_receipt.read_text(encoding="utf-8")),
    )
    if args.gold_summary:
        gold = json.loads(args.gold_summary.read_text(encoding="utf-8"))
        result["gold_reference"] = {
            "native_to_recipient": float(
                gold["cells"]["native_to_recipient"]["semantic_equivalence_rate"]
            ),
            "swap_to_donor": float(
                gold["cells"]["swap_to_donor"]["semantic_equivalence_rate"]
            ),
            "random_matches_either_stable_target_rate": float(
                gold["random_matches_either_stable_target_rate"]
            ),
            "semantic_D_specific": float(gold["semantic_D_specific"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
