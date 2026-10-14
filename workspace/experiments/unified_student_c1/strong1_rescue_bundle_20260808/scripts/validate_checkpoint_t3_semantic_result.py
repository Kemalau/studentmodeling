#!/usr/bin/env python3
"""Validate the fixed 75-row checkpoint T3 semantic diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from validate_t3_semantic_result import (
    MIN_RETENTION_RATIO,
    REFERENCE_SEMANTIC_ACCURACY,
)


EXPECTED_SUPPORT = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 25,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 25,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 25,
}

# The checkpoint panel is stratified to 25 rows per group for bounded cost, but
# the formal panel is not balanced: real Eedi contributes 894/1494 examples.
# Score the diagnostic with the frozen formal-panel prevalences so strong
# synthetic Malrule accuracy cannot hide a collapse on the real Eedi group.
FORMAL_PANEL_SUPPORT = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 894,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 300,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 300,
}
REFERENCE_GROUP_ACCURACY = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 296 / 894,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 156 / 300,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 162 / 300,
}
MIN_GROUP_RETENTION_RATIO = 0.5


def validate(path: Path, *, checkpoint_round: int = 1000) -> dict[str, Any]:
    if checkpoint_round <= 0:
        raise ValueError("checkpoint_round must be positive")
    value = json.loads(path.read_text(encoding="utf-8"))
    groups = value.get("metrics_by_group") or {}
    if set(groups) != set(EXPECTED_SUPPORT):
        raise ValueError(
            f"T3 group mismatch: expected={sorted(EXPECTED_SUPPORT)} got={sorted(groups)}"
        )
    expected_total = sum(EXPECTED_SUPPORT.values())
    if value.get("generation_rows") != expected_total:
        raise ValueError("Checkpoint T3 denominator is not the frozen 75-row panel")
    if value.get("judged_rows") != expected_total:
        raise ValueError("Not every checkpoint T3 generation received a judgment")
    if value.get("judge_errors") != 0:
        raise ValueError("Checkpoint T3 semantic judge has unresolved errors")
    if value.get("judge_backend") not in {"responses", "trapi"}:
        raise ValueError("Unrecognized T3 judge backend")
    if not value.get("judge_rubric_sha256"):
        raise ValueError("T3 judge rubric hash is missing")

    weighted_correct = 0.0
    by_group: dict[str, Any] = {}
    for name, expected in EXPECTED_SUPPORT.items():
        group = groups[name]
        if group.get("n") != expected or group.get("llm_judge_n") != expected:
            raise ValueError(f"Incomplete semantic support for {name}: {group}")
        accuracy = float(group["llm_judge_accuracy"])
        confidence = float(group["llm_judge_mean_confidence"])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"Invalid semantic accuracy for {name}")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid judge confidence for {name}")
        formal_support = FORMAL_PANEL_SUPPORT[name]
        group_retention_ratio = accuracy / REFERENCE_GROUP_ACCURACY[name]
        weighted_correct += accuracy * formal_support
        by_group[name] = {
            "support": expected,
            "formal_panel_support": formal_support,
            "semantic_accuracy": accuracy,
            "reference_semantic_accuracy": REFERENCE_GROUP_ACCURACY[name],
            "retention_ratio": group_retention_ratio,
            "minimum_retention_ratio": MIN_GROUP_RETENTION_RATIO,
            "retention_pass": group_retention_ratio >= MIN_GROUP_RETENTION_RATIO,
            "mean_confidence": confidence,
        }
    semantic_accuracy = weighted_correct / sum(FORMAL_PANEL_SUPPORT.values())
    retention_ratio = semantic_accuracy / REFERENCE_SEMANTIC_ACCURACY
    group_floor_pass = all(row["retention_pass"] for row in by_group.values())
    return {
        "stage": f"layerwise_finalfit_checkpoint{checkpoint_round}_T3_semantic_judge",
        "checkpoint_round": checkpoint_round,
        "diagnostic_only": True,
        "formal_final_evaluation_required": True,
        "pass": retention_ratio >= MIN_RETENTION_RATIO and group_floor_pass,
        "support": expected_total,
        "weighting_scheme": "formal_panel_group_prevalence_v1",
        "formal_panel_support": sum(FORMAL_PANEL_SUPPORT.values()),
        "semantic_accuracy": semantic_accuracy,
        "reference_semantic_accuracy": REFERENCE_SEMANTIC_ACCURACY,
        "retention_ratio": retention_ratio,
        "minimum_retention_ratio": MIN_RETENTION_RATIO,
        "minimum_group_retention_ratio": MIN_GROUP_RETENTION_RATIO,
        "group_floor_pass": group_floor_pass,
        "by_group": by_group,
        "judge_backend": value["judge_backend"],
        "judge_effective_model": value.get("judge_effective_model"),
        "judge_instance": value.get("judge_instance"),
        "judge_api_version": value.get("judge_api_version"),
        "judge_rubric": value.get("judge_rubric"),
        "judge_rubric_sha256": value["judge_rubric_sha256"],
        "metrics": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-round", type=int, default=1000)
    args = parser.parse_args()
    result = validate(args.metrics, checkpoint_round=args.checkpoint_round)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
