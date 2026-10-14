#!/usr/bin/env python3
"""Select a complete-task State8 candidate after same-question causal repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from select_state8_strong1_widen8_bridge import task_mean, task_weighted
except ImportError:  # package-style tests
    from scripts.select_state8_strong1_widen8_bridge import task_mean, task_weighted


ARMS = {
    "updater": {
        "job": "state8-eedi-same-question-updater-mse1-seed3072",
        "seed": 3072,
        "remote_root": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed3072_taskanchor_"
            "source_anchor_updater_mse1_eedi_same_question_v1"
        ),
        "finalfit_variant": "same_question_updater_mse1_seed3072",
    },
    "full": {
        "job": "state8-eedi-same-question-full-mse1-seed3071",
        "seed": 3071,
        "remote_root": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed3071_taskanchor_"
            "source_anchor_full_mse1_eedi_same_question_v1"
        ),
        "finalfit_variant": "same_question_full_mse1_seed3071",
    },
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def one_group(groups: dict[str, Any], fragment: str) -> dict[str, Any]:
    values = [row for name, row in groups.items() if fragment in name]
    if len(values) != 1:
        raise ValueError(f"Expected one semantic group for {fragment}: {groups.keys()}")
    return values[0]


def diagnosis_checks(prefix: str, value: dict[str, Any]) -> dict[str, Any]:
    if value.get("semantic_judge_authoritative") is not True:
        raise ValueError(f"{prefix} semantic judge authority drift")
    if value.get("exact_label_role") != "auxiliary_only_not_used_for_gate":
        raise ValueError(f"{prefix} exact-label role drift")
    cells = value.get("cells") or {}
    return {
        f"{prefix}_pass": bool(value.get("pass")),
        f"{prefix}_D_specific": float(value.get("semantic_D_specific", -1)),
        f"{prefix}_swap_to_donor": float(
            (cells.get("swap_to_donor") or {}).get("semantic_equivalence_rate", -1)
        ),
        f"{prefix}_noop_retention": float(value.get("noop_recipient_retention", -1)),
        f"{prefix}_same_rule_retention": float(
            value.get("same_rule_recipient_retention", -1)
        ),
        f"{prefix}_random_stable": float(
            value.get("random_matches_either_stable_target_rate", 1)
        ),
    }


def evaluate(summary: dict[str, Any], arm: str) -> dict[str, Any]:
    metadata = summary.get("candidate_metadata") or {}
    if arm in ARMS:
        candidate = ARMS[arm]
    else:
        if not arm.startswith("repair_"):
            raise ValueError(f"Unknown arm: {arm}")
        required = {
            "job", "seed", "remote_root", "finalfit_variant",
            "repair_branch", "source_anchor_mode",
        }
        if set(metadata) != required:
            raise ValueError(f"Repair candidate metadata drift: {sorted(metadata)}")
        if metadata["finalfit_variant"] != "same_question_repair_promoted":
            raise ValueError("Repair final-fit variant drift")
        if not str(metadata["remote_root"]).startswith(
            "/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed"
        ):
            raise ValueError("Repair candidate root drift")
        candidate = metadata
    if summary.get("stage") not in {
        "state8_eedi_same_question_causal_repair_semantic_summary_v1",
        "state8_eedi_same_question_full_mse1_semantic_summary_v1",
        "state8_eedi_same_question_repair_semantic_summary_v1",
    }:
        raise ValueError("Same-question summary stage drift")
    if summary.get("status") != "complete":
        raise ValueError("Same-question candidate is incomplete")
    if summary.get("semantic_judge_authoritative") is not True:
        raise ValueError("Semantic judge authority drift")
    if summary.get("exact_match_authoritative") is not False:
        raise ValueError("Exact match authority drift")

    formal = summary.get("formal_metrics") or {}
    formal_groups = formal.get("metrics_by_group") or {}
    semantic = summary.get("t3_semantic") or {}
    if semantic.get("judge_backend") != "trapi" or int(
        semantic.get("judge_errors", -1)
    ) != 0:
        raise ValueError("T3 semantic judgment is incomplete")
    groups = semantic.get("metrics_by_group") or {}
    eedi = one_group(groups, "eedi_mining/open_taxonomy_generation")
    synthetic_cross = one_group(groups, "malrule/synthetic_cross_template")
    synthetic_same = one_group(groups, "malrule/synthetic_same_template")
    if any(int(row.get("llm_judge_n", -1)) != 25 for row in groups.values()):
        raise ValueError("T3 semantic screen support drift")

    checks: dict[str, Any] = {
        # Use the same class-balanced T1 definition as the curriculum matrix.
        # Accuracy is misleading on the heterogeneous binary panels and made
        # repair-vs-curriculum comparisons depend on selector lineage.
        "T1": task_weighted(
            formal_groups, "predict_correctness/", "macro_f1"
        ),
        "T1_accuracy_diagnostic": task_weighted(
            formal_groups, "predict_correctness/", "accuracy"
        ),
        "T2": task_mean(formal_groups, "predict_response/", "math_equivalence"),
        "T3_eedi_semantic": float(eedi["llm_judge_accuracy"]),
        "T3_synthetic_cross_semantic": float(synthetic_cross["llm_judge_accuracy"]),
        "T3_synthetic_same_semantic": float(synthetic_same["llm_judge_accuracy"]),
        "T3_schema": sum(
            float(row["natural_schema_valid_rate"]) * int(row["llm_judge_n"])
            for row in groups.values()
        )
        / sum(int(row["llm_judge_n"]) for row in groups.values()),
        "T5": task_mean(
            formal_groups,
            "predict_prospective_error/",
            "joint_math_equivalence",
        ),
        "T6": task_mean(formal_groups, "predict_error_persistence/", "accuracy"),
        "T6_macro_f1": task_weighted(
            formal_groups, "predict_error_persistence/", "macro_f1"
        ),
        "T6_positive_recall": task_weighted(
            formal_groups, "predict_error_persistence/", "positive_recall"
        ),
        **diagnosis_checks(
            "malrule_diagnosis_C4", summary.get("malrule_diagnosis_c4") or {}
        ),
        **diagnosis_checks(
            "eedi_same_question_C4",
            summary.get("eedi_same_question_diagnosis_c4") or {},
        ),
    }
    floors: dict[str, Any] = {
        "T1": 0.55,
        "T2": 0.20,
        "T3_eedi_semantic": 0.20,
        "T3_synthetic_cross_semantic": 0.45,
        "T3_synthetic_same_semantic": 0.45,
        "T3_schema": 0.95,
        "T5": 0.35,
        "T6": 0.35,
        "T6_macro_f1": 0.30,
        "T6_positive_recall": "strictly_greater_than_zero",
        "malrule_diagnosis_C4": "semantic_gate_pass",
        "eedi_same_question_C4": "semantic_gate_pass",
    }
    numeric = (
        "T1",
        "T2",
        "T3_eedi_semantic",
        "T3_synthetic_cross_semantic",
        "T3_synthetic_same_semantic",
        "T3_schema",
        "T5",
        "T6",
        "T6_macro_f1",
    )
    failed_checks = [name for name in numeric if checks[name] < floors[name]]
    if checks["T6_positive_recall"] <= 0:
        failed_checks.append("T6_positive_recall")
    for name in ("malrule_diagnosis_C4_pass", "eedi_same_question_C4_pass"):
        if checks[name] is not True:
            failed_checks.append(name)
    qualified = not failed_checks
    score = (
        0.10 * checks["T1"]
        + 0.08 * checks["T2"]
        + 0.17 * checks["T3_eedi_semantic"]
        + 0.06 * checks["T3_synthetic_cross_semantic"]
        + 0.06 * checks["T3_synthetic_same_semantic"]
        + 0.08 * checks["T5"]
        + 0.08 * checks["T6_macro_f1"]
        + 0.10 * max(0.0, checks["malrule_diagnosis_C4_D_specific"])
        + 0.10 * max(0.0, checks["eedi_same_question_C4_D_specific"])
        + 0.06 * checks["malrule_diagnosis_C4_swap_to_donor"]
        + 0.08 * checks["eedi_same_question_C4_swap_to_donor"]
        - 0.05 * checks["malrule_diagnosis_C4_random_stable"]
        - 0.08 * checks["eedi_same_question_C4_random_stable"]
    )
    artifact = (
        (summary.get("malrule_diagnosis_c4") or {}).get("generation") or {}
    ).get("artifact_sha256") or {}
    if set(artifact) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Candidate artifact hash drift")
    return {
        "arm": arm,
        "qualified": qualified,
        "failed_checks": failed_checks,
        "score": score,
        "checks": checks,
        "floors": floors,
        "artifact_sha256": artifact,
        **candidate,
    }


def select(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not summaries or any(
        arm not in ARMS and not arm.startswith("repair_") for arm in summaries
    ):
        raise ValueError(f"Candidate matrix drift: {sorted(summaries)}")
    candidates = {arm: evaluate(summaries[arm], arm) for arm in sorted(summaries)}
    eligible = [arm for arm, row in candidates.items() if row["qualified"]]
    selected_arm = max(eligible, key=lambda arm: (candidates[arm]["score"], arm)) if eligible else None
    selected = candidates[selected_arm] if selected_arm else None
    return {
        "stage": "state8_eedi_same_question_joint_diagnostic_selection_v1",
        "status": "complete",
        "pass": selected is not None,
        "selected": selected,
        "selected_arm": selected_arm,
        "candidates": candidates,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "answer_C4_confirmation_pending": selected is not None,
        "full_exact_epoch_pending": selected is not None,
        "selection_scope": (
            "T1/T2/T3/T5/T6 retention plus MalRule and identical-query Eedi "
            "diagnosis-C4; frozen answer-C4 remains a separate gate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("ARM", "SUMMARY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries: dict[str, dict[str, Any]] = {}
    for arm, path in args.candidate:
        if arm in summaries:
            raise ValueError(f"Duplicate arm: {arm}")
        summaries[arm] = read(Path(path))
    result = select(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
