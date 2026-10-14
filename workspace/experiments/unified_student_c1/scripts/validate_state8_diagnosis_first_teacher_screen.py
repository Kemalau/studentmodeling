#!/usr/bin/env python3
"""Semantic gate for the 75-row diagnosis-first teacher preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "diagnose_misconception/eedi_mining/open_taxonomy_generation": (25, 0.35),
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": (25, 0.50),
    "diagnose_misconception/malrule/synthetic_same_template_open_generation": (25, 0.50),
}


def validate(metrics: Path) -> dict:
    value = json.loads(metrics.read_text(encoding="utf-8"))
    if value.get("judge_backend") != "trapi" or value.get("judge_errors") != 0:
        raise ValueError("Authoritative teacher screen requires error-free TRAPI judgment")
    if value.get("generation_rows") != 75 or value.get("judged_rows") != 75:
        raise ValueError("Diagnosis-first teacher screen support drift")
    groups = value.get("metrics_by_group") or {}
    if set(groups) != set(EXPECTED):
        raise ValueError("Diagnosis-first teacher screen group drift")
    output = {}
    for name, (support, threshold) in EXPECTED.items():
        row = groups[name]
        if row.get("n") != support or row.get("llm_judge_n") != support:
            raise ValueError(f"Incomplete teacher group: {name}")
        if row.get("llm_judge_api_or_cache") != support:
            raise ValueError(f"Non-semantic shortcut in teacher group: {name}")
        accuracy = float(row["llm_judge_accuracy"])
        format_valid = float(row["format_valid_rate"])
        schema_valid = float(row["natural_schema_valid_rate"])
        output[name] = {
            "support": support,
            "semantic_accuracy": accuracy,
            "minimum_semantic_accuracy": threshold,
            "format_valid_rate": format_valid,
            "natural_schema_valid_rate": schema_valid,
            "pass": accuracy >= threshold and format_valid >= 0.95 and schema_valid >= 0.95,
        }
    return {
        "stage": "state8_diagnosis_first_full_history_teacher_screen_v1",
        "status": "complete",
        "pass": all(row["pass"] for row in output.values()),
        "support": 75,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "teacher_history_visible": True,
        "student_history_visible_at_final_inference": False,
        "diagnosis_field_order": ["diagnosis", "evidence", "label"],
        "canonical_label_role": "auxiliary_supervision_and_reporting",
        "by_group": output,
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
