#!/usr/bin/env python3
"""Select a State8 repair arm from semantic T3 and diagnosis-C4 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return value


def score(t3: dict, c4: dict) -> tuple[float, dict]:
    groups = t3.get("metrics_by_group") or {}
    expected = {
        "diagnose_misconception/eedi_mining/open_taxonomy_generation": 100,
        "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 100,
        "diagnose_misconception/malrule/synthetic_same_template_open_generation": 100,
    }
    if t3.get("judge_backend") != "trapi" or t3.get("judge_errors") != 0:
        raise ValueError("T3 requires complete authoritative TRAPI judgments")
    if set(groups) != set(expected):
        raise ValueError(f"T3 group drift: {sorted(groups)}")
    correct = 0.0
    schema = 0.0
    for name, support in expected.items():
        row = groups[name]
        if row.get("n") != support or row.get("llm_judge_n") != support:
            raise ValueError(f"Incomplete T3 support: {name}")
        correct += float(row["llm_judge_accuracy"]) * support
        schema += float(row["natural_schema_valid_rate"]) * support
    t3_accuracy = correct / sum(expected.values())
    schema_rate = schema / sum(expected.values())
    if c4.get("semantic_judge_authoritative") is not True:
        raise ValueError("C4 requires authoritative semantic judgments")
    if c4.get("generation", {}).get("num_pairs") != 50:
        raise ValueError("C4 support drift")
    d_specific = float(c4["semantic_D_specific"])
    swap_to_donor = float(c4["cells"]["swap_to_donor"]["semantic_equivalence_rate"])
    noop = float(c4["noop_recipient_retention"])
    same_rule = float(c4["same_rule_recipient_retention"])
    random_any = float(c4["random_matches_either_stable_target_rate"])
    # T3 is the diagnosed collapse, but an arm cannot win by destroying the
    # causal controls.  Keep all terms explicit in the receipt.
    value = (
        0.55 * t3_accuracy
        + 0.20 * max(0.0, d_specific)
        + 0.10 * swap_to_donor
        + 0.075 * noop
        + 0.075 * same_rule
        - 0.10 * random_any
    )
    checks = {
        "t3_semantic_accuracy": t3_accuracy,
        "natural_schema_valid_rate": schema_rate,
        "semantic_D_specific": d_specific,
        "swap_to_donor": swap_to_donor,
        "noop_recipient_retention": noop,
        "same_rule_recipient_retention": same_rule,
        "random_matches_either": random_any,
    }
    return value, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-t3", type=Path, required=True)
    parser.add_argument("--logits-c4", type=Path, required=True)
    parser.add_argument("--trajectory-t3", type=Path, required=True)
    parser.add_argument("--trajectory-c4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {
        "logits": (load(args.logits_t3), load(args.logits_c4)),
        "trajectory01": (load(args.trajectory_t3), load(args.trajectory_c4)),
    }
    results = {}
    for name, (t3, c4) in arms.items():
        value, checks = score(t3, c4)
        results[name] = {"score": value, "checks": checks}
    selected = max(results, key=lambda name: (results[name]["score"], name == "logits"))
    # A repair must materially outperform the collapsed primary partial panel
    # (~0.12 weighted) and retain a non-trivial state intervention effect.
    qualified = (
        results[selected]["checks"]["t3_semantic_accuracy"] >= 0.25
        and results[selected]["checks"]["semantic_D_specific"] >= 0.10
        and results[selected]["checks"]["natural_schema_valid_rate"] >= 0.95
    )
    output = {
        "stage": "state8_stateonly_repair_ab_selection_v1",
        "pass": qualified,
        "selected_arm": selected if qualified else None,
        "arms": results,
        "selection_rule": "0.55*T3 + causal controls; T3>=0.25,D_specific>=0.10,schema>=0.95",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    raise SystemExit(0 if qualified else 2)


if __name__ == "__main__":
    main()
