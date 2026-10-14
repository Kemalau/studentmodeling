#!/usr/bin/env python3
"""Fail closed unless the independent full-history T3 teacher is competent."""

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
MIN_ACCURACY = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": 0.35,
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 0.50,
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": 0.50,
}
MIN_FORMAT_VALID = 0.95
MIN_NATURAL_SCHEMA_VALID = 0.95


def validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("judge_backend") != "trapi":
        raise ValueError("Authoritative teacher judgment requires TRAPI")
    groups = value.get("metrics_by_group") or {}
    if set(groups) != set(EXPECTED_SUPPORT):
        raise ValueError("Teacher semantic group support drift")
    if value.get("generation_rows") != sum(EXPECTED_SUPPORT.values()):
        raise ValueError("Teacher gate is not the frozen 1494-row panel")
    if value.get("judged_rows") != value.get("generation_rows") or value.get("judge_errors") != 0:
        raise ValueError("Teacher semantic judgments are incomplete")
    rows: dict[str, Any] = {}
    for name, support in EXPECTED_SUPPORT.items():
        group = groups[name]
        if group.get("n") != support or group.get("llm_judge_n") != support:
            raise ValueError(f"Incomplete teacher group: {name}")
        # Exact matching is deliberately descriptive only. Every accepted row
        # must have an API/cache semantic judgment from the independent judge.
        if int(group.get("llm_judge_api_or_cache", -1)) != support:
            raise ValueError(f"Teacher group used a non-semantic shortcut: {name}")
        if int(group.get("llm_judge_exact_shortcuts", -1)) != 0:
            raise ValueError(f"Teacher group used exact-match authority: {name}")
        accuracy = float(group["llm_judge_accuracy"])
        format_valid = float(group["format_valid_rate"])
        natural_schema_valid = float(group["natural_schema_valid_rate"])
        if not all(
            math.isfinite(item)
            for item in (accuracy, format_valid, natural_schema_valid)
        ):
            raise ValueError(f"Non-finite teacher metric: {name}")
        rows[name] = {
            "support": support,
            "semantic_accuracy": accuracy,
            "minimum_semantic_accuracy": MIN_ACCURACY[name],
            "format_valid_rate": format_valid,
            "minimum_format_valid_rate": MIN_FORMAT_VALID,
            "natural_schema_valid_rate": natural_schema_valid,
            "minimum_natural_schema_valid_rate": MIN_NATURAL_SCHEMA_VALID,
            "pass": accuracy >= MIN_ACCURACY[name]
            and format_valid >= MIN_FORMAT_VALID
            and natural_schema_valid >= MIN_NATURAL_SCHEMA_VALID,
        }
    passed = all(row["pass"] for row in rows.values())
    return {
        "stage": "state8_independent_full_history_teacher_gate_v1",
        "pass": passed,
        "support": sum(EXPECTED_SUPPORT.values()),
        "exact_match_is_authoritative": False,
        "canonical_label_role": "auxiliary_supervision_and_reporting",
        "teacher_history_visible": True,
        "student_history_visible_at_final_inference": False,
        "by_group": rows,
        "judge_backend": value.get("judge_backend"),
        "judge_effective_model": value.get("judge_effective_model"),
        "judge_rubric_sha256": value.get("judge_rubric_sha256"),
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
