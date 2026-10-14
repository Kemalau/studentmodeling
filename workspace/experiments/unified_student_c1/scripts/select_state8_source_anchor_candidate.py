#!/usr/bin/env python3
"""Select a source-anchor arm for the separate frozen answer-C4 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from select_state8_strong1_widen8_bridge import task_mean, task_weighted
except ImportError:
    from scripts.select_state8_strong1_widen8_bridge import task_mean, task_weighted


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def evaluate(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("stage") != "state8_source_anchor_candidate_semantic_summary_v1":
        raise ValueError("Candidate summary stage drift")
    if summary.get("status") != "complete":
        raise ValueError("Candidate summary is incomplete")
    if summary.get("state_token_count") != 8:
        raise ValueError("State-token count drift")
    if summary.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Decoder history leakage")
    if summary.get("final_inference_route") != "strict_state_only":
        raise ValueError("Final inference route drift")
    if summary.get("semantic_judge_authoritative") is not True:
        raise ValueError("Semantic judge authority drift")
    if summary.get("exact_match_authoritative") is not False:
        raise ValueError("Exact-match authority drift")

    formal = summary.get("formal_metrics") or {}
    groups = summary.get("t3_metrics_by_group") or {}
    diagnosis = summary.get("diagnosis_c4") or {}
    eedi = [row for name, row in groups.items() if "/eedi_mining/" in name]
    malrule = [row for name, row in groups.items() if "/malrule/" in name]
    if len(eedi) != 1 or len(malrule) != 2:
        raise ValueError("T3 semantic group drift")
    checks = {
        "T1": task_mean(formal, "predict_correctness/", "accuracy"),
        "T2": task_mean(formal, "predict_response/", "math_equivalence"),
        "T3_eedi_semantic": float(eedi[0]["llm_judge_accuracy"]),
        "T3_synthetic_semantic": sum(float(row["llm_judge_accuracy"]) for row in malrule) / 2,
        "T3_schema": sum(float(row["natural_schema_valid_rate"]) for row in groups.values()) / 3,
        "T5": task_mean(formal, "predict_prospective_error/", "joint_math_equivalence"),
        "T6": task_mean(formal, "predict_error_persistence/", "accuracy"),
        "T6_macro_f1": task_weighted(formal, "predict_error_persistence/", "macro_f1"),
        "T6_positive_recall": task_weighted(formal, "predict_error_persistence/", "positive_recall"),
        "diagnosis_C4_D_specific": float(diagnosis.get("semantic_D_specific", -1)),
        "diagnosis_C4_swap_to_donor": float(
            ((diagnosis.get("cells") or {}).get("swap_to_donor") or {}).get(
                "semantic_equivalence_rate", -1
            )
        ),
        "diagnosis_C4_noop_retention": float(diagnosis.get("noop_recipient_retention", -1)),
        "diagnosis_C4_same_rule_retention": float(diagnosis.get("same_rule_recipient_retention", -1)),
        "diagnosis_C4_random_stable": float(diagnosis.get("random_matches_either_stable_target_rate", 1)),
    }
    floors: dict[str, Any] = {
        "T1": 0.55,
        "T2": 0.20,
        "T3_eedi_semantic": 0.20,
        "T3_synthetic_semantic": 0.45,
        "T3_schema": 0.95,
        "T5": 0.35,
        "T6": 0.35,
        "T6_macro_f1": 0.30,
        "T6_positive_recall": "strictly_greater_than_zero",
        "diagnosis_C4_D_specific": 0.15,
        "diagnosis_C4_swap_to_donor": 0.50,
        "diagnosis_C4_noop_retention": 0.75,
        "diagnosis_C4_same_rule_retention": 0.65,
        "diagnosis_C4_random_stable_max": 0.25,
    }
    qualified = (
        bool(diagnosis.get("pass"))
        and all(
            checks[key] >= floors[key]
            for key in (
                "T1", "T2", "T3_eedi_semantic", "T3_synthetic_semantic",
                "T3_schema", "T5", "T6", "T6_macro_f1",
                "diagnosis_C4_D_specific", "diagnosis_C4_swap_to_donor",
                "diagnosis_C4_noop_retention", "diagnosis_C4_same_rule_retention",
            )
        )
        and checks["T6_positive_recall"] > 0
        and checks["diagnosis_C4_random_stable"] <= floors["diagnosis_C4_random_stable_max"]
    )
    score = (
        0.12 * checks["T1"] + 0.10 * checks["T2"]
        + 0.20 * checks["T3_eedi_semantic"] + 0.10 * checks["T3_synthetic_semantic"]
        + 0.10 * checks["T5"] + 0.10 * checks["T6"]
        + 0.13 * max(0.0, checks["diagnosis_C4_D_specific"])
        + 0.08 * checks["diagnosis_C4_swap_to_donor"]
        + 0.035 * checks["diagnosis_C4_noop_retention"]
        + 0.035 * checks["diagnosis_C4_same_rule_retention"]
        - 0.10 * checks["diagnosis_C4_random_stable"]
    )
    return {"diagnostic_qualified": qualified, "score": score, "checks": checks, "floors": floors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates: dict[str, Any] = {}
    for path in args.candidate:
        summary = read(path)
        arm = str(summary.get("arm") or "")
        if not arm or arm in candidates:
            raise ValueError(f"Invalid or duplicate arm: {arm!r}")
        candidates[arm] = {**evaluate(summary), "job": summary.get("job"), "artifact_sha256": summary.get("artifact_sha256")}
    if set(candidates) != {"full_mse1", "updater_mse1", "full_mse2"}:
        raise ValueError(f"Candidate matrix drift: {sorted(candidates)}")
    qualified = [name for name, row in candidates.items() if row["diagnostic_qualified"]]
    selected = max(qualified, key=lambda name: (candidates[name]["score"], name)) if qualified else None
    result = {
        "stage": "state8_source_anchor_diagnostic_selection_v1",
        "status": "complete",
        "diagnostic_gate_pass": selected is not None,
        "selected_for_answer_c4_confirmation": selected,
        "scientific_gate_pass": False,
        "answer_c4_confirmation_pending": selected is not None,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "candidates": candidates,
        "selection_rule": "joint T1/T2/semantic-T3/T5/T6 and diagnosis-C4 gate; separate frozen answer-C4 still required",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if selected is not None else 2)


if __name__ == "__main__":
    main()
