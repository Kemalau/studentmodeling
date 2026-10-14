#!/usr/bin/env python3
"""Rank early same-question checkpoints without mistaking screens for promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def c4_checks(value: dict[str, Any]) -> dict[str, float]:
    cells = value.get("cells") or {}
    return {
        "D_specific": float(value.get("semantic_D_specific", -1)),
        "swap_to_donor": float((cells.get("swap_to_donor") or {}).get("semantic_equivalence_rate", -1)),
        "native_to_recipient": float((cells.get("native_to_recipient") or {}).get("semantic_equivalence_rate", -1)),
        "noop_retention": float(value.get("noop_recipient_retention", -1)),
        "same_rule_retention": float(value.get("same_rule_recipient_retention", -1)),
        "random_stable": float(value.get("random_matches_either_stable_target_rate", 1)),
    }


def evaluate(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("stage") != "state8_eedi_same_question_checkpoint_semantic_summary_v1":
        raise ValueError("Checkpoint summary stage drift")
    if summary.get("status") != "complete" or summary.get("diagnostic_only") is not True:
        raise ValueError("Checkpoint summary is not a diagnostic completion")
    if summary.get("semantic_judge_authoritative") is not True:
        raise ValueError("Semantic judge authority drift")
    if summary.get("exact_match_authoritative") is not False:
        raise ValueError("Exact-match authority drift")
    arm = str(summary.get("arm")); step = int(summary.get("checkpoint_step", -1))
    if arm not in {"updater", "full"} or step not in {20, 100}:
        raise ValueError("Arm/step drift")
    malrule = c4_checks(summary.get("malrule_diagnosis_c4") or {})
    eedi = c4_checks(summary.get("eedi_same_question_diagnosis_c4") or {})
    checks = {
        "T3_eedi_semantic": float(summary.get("T3_eedi_semantic", -1)),
        "T3_synthetic_cross_semantic": float(summary.get("T3_synthetic_cross_semantic", -1)),
        "T3_synthetic_same_semantic": float(summary.get("T3_synthetic_same_semantic", -1)),
        **{f"malrule_{key}": value for key, value in malrule.items()},
        **{f"eedi_{key}": value for key, value in eedi.items()},
    }
    hard_sanity = (
        checks["T3_eedi_semantic"] >= 0.20
        and min(checks["T3_synthetic_cross_semantic"], checks["T3_synthetic_same_semantic"]) >= 0.40
        and checks["malrule_random_stable"] <= 0.40
        and checks["eedi_random_stable"] <= 0.40
        and checks["eedi_D_specific"] >= 0.10
        and checks["eedi_swap_to_donor"] >= 0.40
        and checks["eedi_noop_retention"] >= 0.60
        and checks["eedi_same_rule_retention"] >= 0.60
    )
    score = (
        0.28 * checks["T3_eedi_semantic"]
        + 0.08 * checks["T3_synthetic_cross_semantic"]
        + 0.08 * checks["T3_synthetic_same_semantic"]
        + 0.10 * max(0.0, checks["malrule_D_specific"])
        + 0.08 * checks["malrule_swap_to_donor"]
        + 0.16 * max(0.0, checks["eedi_D_specific"])
        + 0.10 * checks["eedi_swap_to_donor"]
        + 0.04 * checks["eedi_native_to_recipient"]
        + 0.04 * checks["eedi_same_rule_retention"]
        + 0.04 * checks["eedi_noop_retention"]
        - 0.06 * checks["malrule_random_stable"]
        - 0.10 * checks["eedi_random_stable"]
    )
    failed_sanity = []
    for name, passed in (
        ("T3_eedi_semantic", checks["T3_eedi_semantic"] >= 0.20),
        ("T3_synthetic_semantic", min(
            checks["T3_synthetic_cross_semantic"],
            checks["T3_synthetic_same_semantic"],
        ) >= 0.40),
        ("malrule_random_stable", checks["malrule_random_stable"] <= 0.40),
        ("eedi_random_stable", checks["eedi_random_stable"] <= 0.40),
        ("eedi_D_specific", checks["eedi_D_specific"] >= 0.10),
        ("eedi_swap_to_donor", checks["eedi_swap_to_donor"] >= 0.40),
        ("eedi_noop_retention", checks["eedi_noop_retention"] >= 0.60),
        ("eedi_same_rule_retention", checks["eedi_same_rule_retention"] >= 0.60),
    ):
        if not passed:
            failed_sanity.append(name)
    return {
        "arm": arm,
        "step": step,
        "checks": checks,
        "hard_sanity": hard_sanity,
        "failed_sanity": failed_sanity,
        "score": score,
    }


def select(values: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [evaluate(value) for value in values]
    keys = {(row["arm"], row["step"]) for row in candidates}
    expected = {(arm, step) for arm in ("updater", "full") for step in (20, 100)}
    if keys != expected:
        raise ValueError(f"Checkpoint matrix drift: {sorted(keys)}")
    viable = [row for row in candidates if row["hard_sanity"]]
    selected = max(viable, key=lambda row: (row["score"], row["arm"], -row["step"])) if viable else None
    return {
        "stage": "state8_eedi_same_question_early_checkpoint_selection_v1",
        "status": "complete", "diagnostic_only": True,
        "semantic_judge_authoritative": True, "exact_match_authoritative": False,
        "selected_for_extended_training": selected,
        "candidates": sorted(candidates, key=lambda row: (row["arm"], row["step"])),
        "formal_unified_promotion_pending": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select([read(path) for path in args.candidate])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["selected_for_extended_training"] else 2)


if __name__ == "__main__":
    main()
