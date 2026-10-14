#!/usr/bin/env python3
"""Validate fixed-denominator semantic T3 judging for the final recurrent model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_SUPPORT = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 894,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 300,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 300,
}
REFERENCE_SEMANTIC_ACCURACY = 0.4109772423025435
MIN_RETENTION_RATIO = 0.85
REFERENCE_GROUP_ACCURACY = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 296 / 894,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 156 / 300,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 162 / 300,
}
MIN_GROUP_RETENTION_RATIO = 0.5
MIN_NATURAL_SCHEMA_VALID = 0.95


def validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    groups = value.get("metrics_by_group") or {}
    if set(groups) != set(EXPECTED_SUPPORT):
        raise ValueError(
            f"T3 group mismatch: expected={sorted(EXPECTED_SUPPORT)} got={sorted(groups)}"
        )
    if value.get("generation_rows") != sum(EXPECTED_SUPPORT.values()):
        raise ValueError("T3 generation denominator is not the frozen 1494-row panel")
    if value.get("judged_rows") != value.get("generation_rows"):
        raise ValueError("Not every T3 generation received a semantic judgment")
    if value.get("judge_errors") != 0:
        raise ValueError("T3 semantic judge has unresolved errors")
    if value.get("judge_backend") != "trapi":
        raise ValueError("Authoritative T3 semantic judgment requires TRAPI")
    if not value.get("judge_rubric_sha256"):
        raise ValueError("T3 judge rubric hash is missing")

    weighted_correct = 0.0
    total = 0
    by_group: dict[str, Any] = {}
    for name, expected in EXPECTED_SUPPORT.items():
        group = groups[name]
        if group.get("n") != expected or group.get("llm_judge_n") != expected:
            raise ValueError(f"Incomplete semantic support for {name}: {group}")
        if int(group.get("llm_judge_api_or_cache", -1)) != expected:
            raise ValueError(f"T3 group lacks independent semantic judgments: {name}")
        if int(group.get("llm_judge_exact_shortcuts", -1)) != 0:
            raise ValueError(f"T3 group used exact-match authority: {name}")
        accuracy = float(group["llm_judge_accuracy"])
        confidence = float(group["llm_judge_mean_confidence"])
        natural_schema_valid = float(group["natural_schema_valid_rate"])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"Invalid T3 semantic accuracy for {name}")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid T3 judge confidence for {name}")
        if not math.isfinite(natural_schema_valid) or not 0.0 <= natural_schema_valid <= 1.0:
            raise ValueError(f"Invalid T3 natural schema rate for {name}")
        group_retention_ratio = accuracy / REFERENCE_GROUP_ACCURACY[name]
        by_group[name] = {
            "support": expected,
            "semantic_accuracy": accuracy,
            "reference_semantic_accuracy": REFERENCE_GROUP_ACCURACY[name],
            "retention_ratio": group_retention_ratio,
            "minimum_retention_ratio": MIN_GROUP_RETENTION_RATIO,
            "retention_pass": group_retention_ratio >= MIN_GROUP_RETENTION_RATIO,
            "mean_confidence": confidence,
            "natural_schema_valid_rate": natural_schema_valid,
            "minimum_natural_schema_valid_rate": MIN_NATURAL_SCHEMA_VALID,
            "natural_schema_pass": natural_schema_valid >= MIN_NATURAL_SCHEMA_VALID,
        }
        weighted_correct += accuracy * expected
        total += expected
    semantic_accuracy = weighted_correct / total
    retention_ratio = semantic_accuracy / REFERENCE_SEMANTIC_ACCURACY
    group_floor_pass = all(row["retention_pass"] for row in by_group.values())
    natural_schema_pass = all(row["natural_schema_pass"] for row in by_group.values())
    return {
        "stage": "final_T3_semantic_judge",
        "pass": retention_ratio >= MIN_RETENTION_RATIO
        and group_floor_pass
        and natural_schema_pass,
        "support": total,
        "weighting_scheme": "formal_panel_group_prevalence_v1",
        "semantic_accuracy": semantic_accuracy,
        "reference_semantic_accuracy": REFERENCE_SEMANTIC_ACCURACY,
        "retention_ratio": retention_ratio,
        "minimum_retention_ratio": MIN_RETENTION_RATIO,
        "minimum_group_retention_ratio": MIN_GROUP_RETENTION_RATIO,
        "group_floor_pass": group_floor_pass,
        "natural_schema_pass": natural_schema_pass,
        "minimum_natural_schema_valid_rate": MIN_NATURAL_SCHEMA_VALID,
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
    args = parser.parse_args()
    result = validate(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
