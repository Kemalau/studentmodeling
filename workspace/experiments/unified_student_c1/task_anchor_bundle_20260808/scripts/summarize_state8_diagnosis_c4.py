#!/usr/bin/env python3
"""Combine State8 diagnosis-C4 generations with independent semantic judgments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from t3_judge_utils import read_jsonl
except ImportError:  # package-style unit tests
    from scripts.t3_judge_utils import read_jsonl


def summarize(judged: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[int, str], dict[str, bool]] = defaultdict(dict)
    for row in judged:
        condition = str(row["condition"])
        role = str(row["reference_role"])
        grouped[(condition, role)].append(row)
        by_pair[(int(row["pair_idx"]), condition)][role] = bool(
            row["llm_judge_correct"]
        )
    expected = {(condition, role) for condition in ("native", "noop", "samemal", "random", "swap") for role in ("donor", "recipient")}
    if set(grouped) != expected:
        raise ValueError(f"Semantic diagnosis-C4 cell drift: {sorted(grouped)}")
    cells = {
        f"{condition}_to_{role}": {
            "n": len(rows),
            "semantic_equivalence_rate": sum(
                bool(row["llm_judge_correct"]) for row in rows
            )
            / len(rows),
            "mean_confidence": sum(float(row["llm_judge_confidence"]) for row in rows)
            / len(rows),
        }
        for (condition, role), rows in sorted(grouped.items())
    }
    random_any = sum(
        values.get("donor", False) or values.get("recipient", False)
        for (pair_idx, condition), values in by_pair.items()
        if condition == "random"
    ) / len(grouped[("random", "donor")])
    return {
        "stage": "state8_closed_diagnosis_c4_semantic_smoke_v1",
        "exploratory_only": True,
        "cells": cells,
        "semantic_D_specific": (
            cells["swap_to_donor"]["semantic_equivalence_rate"]
            - cells["random_to_donor"]["semantic_equivalence_rate"]
        ),
        "random_matches_either_stable_target_rate": random_any,
        "noop_recipient_retention": cells["noop_to_recipient"]["semantic_equivalence_rate"],
        "same_rule_recipient_retention": cells["samemal_to_recipient"]["semantic_equivalence_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-redacted", type=Path, required=True)
    parser.add_argument("--generation-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    judged = read_jsonl(args.judge_redacted)
    if not judged or any("llm_judge_correct" not in row for row in judged):
        raise ValueError("Diagnosis-C4 semantic panel is incomplete")
    generation = json.loads(args.generation_metrics.read_text(encoding="utf-8"))
    result = summarize(judged)
    result["generation"] = generation
    result["semantic_judge_authoritative"] = True
    result["exact_label_role"] = "auxiliary_only_not_used_for_gate"
    result["pass"] = (
        result["semantic_D_specific"] >= 0.15
        and result["random_matches_either_stable_target_rate"] <= 0.25
        and result["noop_recipient_retention"] >= 0.75
        and result["same_rule_recipient_retention"] >= 0.65
        and result["cells"]["swap_to_donor"]["semantic_equivalence_rate"] >= 0.50
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
