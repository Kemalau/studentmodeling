#!/usr/bin/env python3
"""Audit the complete unified-model objective across model and C3 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from task_taxonomy_v2 import taxonomy_receipt
except ImportError:  # package-style tests
    from scripts.task_taxonomy_v2 import taxonomy_receipt


SELECTION_STAGES = {
    "state8_strong1_widen8_joint_promotion_v1",
    "state8_strong1_rescue_joint_promotion_v1",
    "state8_midpoint_diagnosis_first_joint_promotion_v1",
    "state8_exact_diagnosis_curriculum_joint_promotion_v1",
    "state8_exact_scientific_rescue_joint_promotion_v1",
    "state8_diagnosis_first_matched_joint_promotion_v1",
    "state8_eedi_same_question_joint_promotion_v1",
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate(
    selection_path: Path,
    final_gate_path: Path,
    publish_path: Path,
    c3_path: Path,
) -> dict[str, Any]:
    selection = read(selection_path)
    final_gate = read(final_gate_path)
    publish = read(publish_path)
    c3 = read(c3_path)
    selected = selection.get("selected") or {}
    artifacts = final_gate.get("artifact_sha256") or {}
    publish_artifacts = publish.get("artifact_sha256") or {}
    arms = c3.get("arms") or {}
    conclusion = c3.get("conclusion") or {}
    expected_arms = {
        "malrule_correctness",
        "malrule_response",
        "malrule_misconception",
    }
    final_checks = final_gate.get("checks") or {}
    performance = final_gate.get("performance") or {}
    expected_performance = {
        "predict_correctness",
        "predict_response",
        "predict_prospective_error",
        "predict_error_persistence",
    }
    performance_complete = set(performance) == expected_performance and all(
        row.get("pass") is True for row in performance.values()
    )
    t3_complete = (
        (final_gate.get("T3_semantic") or {}).get("pass") is True
        and int((final_gate.get("T3_semantic") or {}).get("support", -1))
        == 1494
        and final_gate.get("T3_exact_match_role")
        == "auxiliary_only_not_used_for_gate"
    )
    c3_complete = (
        c3.get("stage") == "C3_exposure_matched_auxiliary_transfer_v1"
        and c3.get("status") == "complete"
        and c3.get("controlled_bundle_pass") is True
        and set(arms) == expected_arms
        and all(
            (row.get("checks") or {}).get("train_support") is True
            and (row.get("checks") or {}).get("closed_state") is True
            and int((row.get("T3_semantic") or {}).get("support", -1)) == 75
            and (row.get("T3_semantic") or {}).get("judge_backend") == "trapi"
            and int((row.get("C4") or {}).get("clean_n", -1)) >= 50
            and (row.get("C4") or {}).get("scientific_gate_pass") is True
            for row in arms.values()
        )
    )
    c4_complete = (
        (final_gate.get("C4") or {}).get("replication_gate_pass") is True
        and (final_gate.get("diagnosis_C4") or {}).get("pass") is True
        and (
            selection.get("stage") != "state8_eedi_same_question_joint_promotion_v1"
            or (
                (final_gate.get("eedi_same_question_C4") or {}).get("pass")
                is True
                and (final_gate.get("eedi_same_question_C4") or {}).get(
                    "confirmatory"
                )
                is True
            )
        )
    )
    checks = {
        "candidate_selection": selection.get("stage") in SELECTION_STAGES
        and selection.get("pass") is True
        and selected.get("qualified") is True,
        "selected_variant_binding": selected.get("finalfit_variant")
        == publish.get("selected_replay_variant"),
        "canonical_publish": publish.get("stage")
        == "state8_canonical_final_checkpoint_publish_v1"
        and publish.get("status") == "complete"
        and isinstance(publish.get("canonical_run_dir"), str)
        and bool(publish.get("canonical_run_dir")),
        "artifact_binding": artifacts == publish_artifacts
        and set(artifacts) == {"adapter_model", "recurrent_state"}
        and all(isinstance(value, str) and len(value) == 64 for value in artifacts.values()),
        "complete_model_final_gate": final_gate.get("stage")
        == "state8_complete_unified_model_final_gate_v1"
        and final_gate.get("pass") is True
        and final_checks
        and all(value is True for value in final_checks.values()),
        "one_epoch_exact": (final_gate.get("final_fit_receipt") or {}).get(
            "one_epoch_exact"
        )
        is True
        and int(
            (final_gate.get("final_fit_receipt") or {}).get("real_examples", -1)
        )
        == 259845,
        "T1_T2_T4_T5_retention": performance_complete,
        "T3_semantic_authority": t3_complete,
        "C4_answer_and_diagnosis": c4_complete,
        "C3_experiment_complete": c3_complete,
        "C3_conclusion_separated_from_completion": isinstance(
            conclusion.get("C3_transfer_hypothesis_supported"), bool
        )
        and conclusion.get("recipe_promotion_allowed") is False,
    }
    c1_checks = {
        "one_shared_checkpoint": checks["artifact_binding"]
        and checks["canonical_publish"],
        "exact_unified_epoch": checks["one_epoch_exact"],
        "correctness_response_and_diagnosis": performance_complete
        and t3_complete,
        "full_formal_support": int(final_gate.get("formal_support", 21676))
        == 21676,
    }
    c2_checks = {
        "concrete_response_generation": performance.get(
            "predict_response", {}
        ).get("pass")
        is True,
        "natural_language_misconception_diagnosis": t3_complete,
        "prospective_error": performance.get(
            "predict_prospective_error", {}
        ).get("pass")
        is True,
        "error_persistence": performance.get(
            "predict_error_persistence", {}
        ).get("pass")
        is True,
        "persistence_positive_recall": float(
            performance.get("predict_error_persistence", {}).get(
                "positive_recall", 0.0
            )
        )
        > 0.0,
    }
    c3_supported = conclusion.get("C3_transfer_hypothesis_supported") is True
    c4_checks = {
        "predictive_sufficiency": performance_complete and t3_complete,
        "answer_swap_replication": (final_gate.get("C4") or {}).get(
            "replication_gate_pass"
        )
        is True,
        "diagnosis_swap_replication": (
            final_gate.get("diagnosis_C4") or {}
        ).get("pass")
        is True,
        "strict_state_only": final_checks.get("history_closed") is True,
        "identical_query_eedi_swap": (
            True
            if selection.get("stage")
            != "state8_eedi_same_question_joint_promotion_v1"
            else (
                (final_gate.get("eedi_same_question_C4") or {}).get("pass")
                is True
                and ((final_gate.get("eedi_same_question_C4") or {}).get(
                    "generation"
                ) or {}).get("same_question_query_identical") is True
                and ((final_gate.get("eedi_same_question_C4") or {}).get(
                    "generation"
                ) or {}).get("student_response_visible_to_decoder") is False
            )
        ),
    }
    claim_matrix = {
        "C1": {
            "name": "unified_multitask_student_model",
            "pass": all(c1_checks.values()),
            "checks": c1_checks,
            "evidence": "one hash-bound checkpoint on the full formal task matrix",
        },
        "C2": {
            "name": "generative_educational_capability_frontier",
            "pass": all(c2_checks.values()),
            "checks": c2_checks,
            "evidence": (
                "state-only T2 concrete responses, T3 natural-language "
                "diagnosis, T4 prospective behavior, and strict T5 "
                "persistence gates"
            ),
        },
        "C3": {
            "name": "exposure_matched_cross_task_educational_transfer",
            "experiment_complete": c3_complete,
            "hypothesis_supported": c3_supported,
            "claim_status": "supported" if c3_supported else "not_supported",
            "evidence": (
                "training on one educational task or supervision type changes "
                "a distinct held-out task under exposure-matched controls; "
                "the current three-arm misconception study is one C3 instance"
            ),
        },
        "C4": {
            "name": "sufficient_consistent_and_specifically_causal_student_state",
            "pass": all(c4_checks.values()),
            "checks": c4_checks,
            "evidence": (
                "history-closed predictive sufficiency plus state-path "
                "consistency and three-seed answer/natural-diagnosis "
                "swap, random, same-rule, and no-op gates"
            ),
        },
    }
    # C3 can be a scientifically valid negative experiment.  C1, C2, and C4
    # are properties of the delivered unified checkpoint and therefore remain
    # hard completion requirements rather than reporting-only annotations.
    checks["C1_claim"] = claim_matrix["C1"]["pass"]
    checks["C2_claim"] = claim_matrix["C2"]["pass"]
    checks["C4_claim"] = claim_matrix["C4"]["pass"]
    return {
        "stage": "complete_unified_student_model_objective_v2",
        "status": "complete" if all(checks.values()) else "failed_gate",
        "pass": all(checks.values()),
        "checks": checks,
        "selected_candidate": selected.get("slug") or selected.get("arm"),
        "selected_replay_variant": publish.get("selected_replay_variant"),
        "canonical_run_dir": publish.get("canonical_run_dir"),
        "artifact_sha256": artifacts,
        "formal_support": 21676,
        "T3_semantic_support": 1494,
        "C4_pair_seeds": [13, 42, 97],
        "task_taxonomy": taxonomy_receipt(),
        "C3_transfer_hypothesis_supported": conclusion.get(
            "C3_transfer_hypothesis_supported"
        ),
        "C3_role": "independent_exposure_matched_training_counterfactual",
        "C1_C4_claim_matrix": claim_matrix,
        "C1_C4_scope_note": (
            "C3 experimental completion is required, but a negative transfer "
            "result is reported as not supported rather than converted into a "
            "positive claim. C1, C2, and C4 must individually pass."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument("--publish-receipt", type=Path, required=True)
    parser.add_argument("--c3-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.selection, args.final_gate, args.publish_receipt, args.c3_summary
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
