#!/usr/bin/env python3
"""Audit and summarize the exposure-matched C3 transfer experiment.

The three arms see identical Eedi rows, identical underlying Malrule examples,
and an identical role order.  Only the Malrule auxiliary target differs.  A
complete controlled experiment is useful even when the C3 transfer hypothesis
is not supported, so completion and a positive scientific conclusion are
reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARMS = {
    "malrule_correctness": "predict_correctness",
    "malrule_response": "predict_response",
    "malrule_misconception": "diagnose_misconception",
}
T6_REFERENCE_MACRO_F1 = 0.5959866676584252
T6_MIN_RATIO = 0.85
MIN_C3_EFFECT = 0.05
EEDI_GROUP = "diagnose_misconception/eedi_mining/open_taxonomy_generation"


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def weighted(groups: dict[str, Any], task: str, metric: str) -> float:
    rows = [row for name, row in groups.items() if name.startswith(task + "/")]
    if not rows or any(metric not in row for row in rows):
        raise ValueError(f"Missing {metric} support for {task}")
    support = sum(int(row["n"]) for row in rows)
    if support <= 0:
        raise ValueError(f"Empty support for {task}")
    return sum(float(row[metric]) * int(row["n"]) for row in rows) / support


def find_arm_roots(root: Path) -> dict[str, Path]:
    matches: dict[str, list[Path]] = {arm: [] for arm in ARMS}
    for manifest_path in root.rglob("task_retention_index_manifest.json"):
        manifest = read(manifest_path)
        arm = str(manifest.get("arm") or "")
        if arm not in matches:
            continue
        for parent in manifest_path.parents:
            if (parent / "eval/c4/recurrent_causal_gate.json").is_file():
                matches[arm].append(parent)
                break
            if parent == root:
                break
    invalid = {arm: paths for arm, paths in matches.items() if len(paths) != 1}
    if invalid:
        raise ValueError(f"Expected exactly one result per C3 arm: {invalid}")
    return {arm: paths[0] for arm, paths in matches.items()}


def controlled_bundle_check(manifests: dict[str, dict[str, Any]]) -> bool:
    shared_keys = (
        "eedi_example_ids_sha256",
        "paired_malrule_raw_ids_sha256",
        "role_order_sha256",
        "start_global_position",
        "segment_rows",
        "source_train_sha256",
        "formal_eval_sha256",
    )
    shared = {tuple(manifest.get(key) for key in shared_keys) for manifest in manifests.values()}
    if len(shared) != 1:
        return False
    for arm, auxiliary_task in ARMS.items():
        manifest = manifests[arm]
        expected_tasks = {"diagnose_misconception": 500, auxiliary_task: 500}
        if auxiliary_task == "diagnose_misconception":
            expected_tasks = {"diagnose_misconception": 1000}
        if (
            manifest.get("stage") != "layerwise_task_retention_index_manifest_v1"
            or manifest.get("controlled_experiment")
            != "C3_exposure_matched_auxiliary_supervision_v1"
            or manifest.get("controlled_factor") != "malrule_auxiliary_target"
            or manifest.get("arm") != arm
            or manifest.get("auxiliary_task") != auxiliary_task
            or manifest.get("eedi_rows") != 500
            or manifest.get("malrule_rows") != 500
            or manifest.get("source_counts") != {"eedi_mining": 500, "malrule": 500}
            or manifest.get("task_counts") != expected_tasks
            or manifest.get("formal_example_id_overlap") != 0
            or manifest.get("same_eedi_rows_across_arms") is not True
            or manifest.get("same_underlying_malrule_rows_across_arms") is not True
            or manifest.get("same_role_order_across_arms") is not True
        ):
            return False
    return True


def transfer_conclusion(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = {
        arm: float(row["T3_semantic"]["eedi_accuracy"])
        for arm, row in rows.items()
    }
    overall = {
        arm: float(row["T3_semantic"]["semantic_accuracy"])
        for arm, row in rows.items()
    }
    controls = ("malrule_correctness", "malrule_response")
    best_control = max(controls, key=lambda arm: (primary[arm], overall[arm]))
    effect = primary["malrule_misconception"] - primary[best_control]
    overall_effect = overall["malrule_misconception"] - overall[best_control]
    all_arms_preserve_non_t3 = all(
        bool(row["non_T3_preservation_pass"]) for row in rows.values()
    )
    misconception_t3_pass = bool(
        rows["malrule_misconception"]["T3_semantic"]["pass"]
    )
    checks = {
        # T3 is the controlled outcome. Requiring both negative-control arms to
        # pass the T3 gate would make a selective positive effect impossible by
        # construction. Every arm must preserve C4 and the non-T3 tasks, while
        # the misconception-treatment arm must additionally pass the T3 gate.
        "all_arms_preserve_non_T3_gates": all_arms_preserve_non_t3,
        "misconception_arm_passes_T3_gate": misconception_t3_pass,
        "misconception_auxiliary_is_best_on_real_eedi_T3": (
            primary["malrule_misconception"] == max(primary.values())
        ),
        "real_eedi_T3_effect_at_least_0.05": effect >= MIN_C3_EFFECT,
        "overall_T3_not_worse_than_best_control": overall_effect >= -0.02,
        "one_training_seed_only_not_promotable": True,
    }
    supported = all(checks.values())
    recommended = max(
        rows,
        key=lambda arm: (
            bool(rows[arm]["non_T3_preservation_pass"]),
            primary[arm],
            overall[arm],
            float(rows[arm]["C4"]["D_specific"]),
        ),
    )
    return {
        "primary_endpoint": EEDI_GROUP + "/llm_judge_accuracy",
        "primary_scores": primary,
        "overall_T3_semantic_scores": overall,
        "best_control": best_control,
        "misconception_minus_best_control_real_eedi_T3": effect,
        "misconception_minus_best_control_overall_T3": overall_effect,
        "checks": checks,
        "C3_transfer_hypothesis_supported": supported,
        "recommended_auxiliary_target_for_replication": recommended,
        "replication_recommended": supported,
        "recipe_promotion_allowed": False,
    }


def summarize(root: Path, t3_root: Path) -> dict[str, Any]:
    arm_roots = find_arm_roots(root)
    manifests: dict[str, dict[str, Any]] = {}
    rows: dict[str, dict[str, Any]] = {}
    for arm, job in sorted(arm_roots.items()):
        manifest = read(next(job.rglob("task_retention_index_manifest.json")))
        receipt = read(next(job.rglob("task_retention_training_receipt.json")))
        completion = read(next(job.rglob("completion.json")))
        config = read(next(job.rglob("run_config.json")))
        formal = read(job / "eval/formal_screen/metrics.json")
        t6 = read(job / "eval/t6_full/metrics.json")
        c4_payload = read(job / "eval/c4/recurrent_causal_gate.json")
        semantic = read(t3_root / arm / "gate.json")
        manifests[arm] = manifest

        formal_groups = formal["metrics_by_group"]
        t6_groups = t6["metrics_by_group"]
        c4 = c4_payload["summary"]
        clean = c4["clean"]
        t6_macro = weighted(t6_groups, "predict_error_persistence", "macro_f1")
        t6_recall = weighted(t6_groups, "predict_error_persistence", "positive_recall")
        formal_scores = {
            "predict_correctness": weighted(formal_groups, "predict_correctness", "macro_f1"),
            "predict_response": weighted(formal_groups, "predict_response", "math_equivalence"),
            "predict_prospective_error": weighted(
                formal_groups, "predict_prospective_error", "joint_math_equivalence"
            ),
            "T3_format_valid_rate": weighted(
                formal_groups, "diagnose_misconception", "format_valid_rate"
            ),
        }
        eedi_semantic = semantic.get("by_group", {}).get(EEDI_GROUP) or {}
        checks = {
            "diagnostic_only": receipt.get("diagnostic_only") is True,
            "source_round_1500": receipt.get("source_checkpoint_round") == 1500,
            "train_support": completion.get("micro_steps") == 1000
            and completion.get("optimizer_steps") == 250,
            "causal_exposure": completion.get("causal_pairs_seen") == 50,
            "state_path_recipe": receipt.get("loss_profile") == "state_path"
            and float(config.get("causal_weight", -1)) == 2.0,
            "closed_state": c4_payload.get("state_architecture") == "layerwise"
            and c4_payload.get("raw_history_visible_to_decoder") is False
            and c4_payload.get("persistent_history_kv_cache") is False,
            "C4": c4.get("scientific_gate_pass") is True,
            "T3_semantic": semantic.get("pass") is True
            and semantic.get("support") == 75
            and eedi_semantic.get("support") == 25,
            "formal_retention": formal_scores["predict_correctness"] >= 0.40
            and formal_scores["predict_response"] >= 0.20
            and formal_scores["predict_prospective_error"] >= 0.40
            and formal_scores["T3_format_valid_rate"] >= 0.80,
            "T6_retention": t6.get("sampled_rows") == 976
            and t6_macro / T6_REFERENCE_MACRO_F1 >= T6_MIN_RATIO
            and t6_recall > 0.0,
        }
        non_t3_checks = {key: value for key, value in checks.items() if key != "T3_semantic"}
        rows[arm] = {
            "arm": arm,
            "auxiliary_task": ARMS[arm],
            "job_root": str(job),
            "checks": checks,
            "all_gates_pass": all(checks.values()),
            "non_T3_preservation_pass": all(non_t3_checks.values()),
            "formal_screen": formal_scores,
            "T6": {
                "support": int(t6["sampled_rows"]),
                "macro_f1": t6_macro,
                "positive_recall": t6_recall,
                "retention_ratio": t6_macro / T6_REFERENCE_MACRO_F1,
            },
            "C4": {
                "clean_n": int(c4["clean_n"]),
                "swap": float(clean["swap"]["flip_to_A_rate"]),
                "random": float(clean["random"]["flip_to_A_rate"]),
                "same_rule": float(clean["samemal"]["flip_to_A_rate"]),
                "noop": float(clean["noop"]["flip_to_A_rate"]),
                "D_specific": float(c4["D_specific"]),
                "scientific_gate_pass": bool(c4["scientific_gate_pass"]),
            },
            "T3_semantic": {
                "support": int(semantic["support"]),
                "semantic_accuracy": float(semantic["semantic_accuracy"]),
                "eedi_accuracy": float(eedi_semantic["semantic_accuracy"]),
                "retention_ratio": float(semantic["retention_ratio"]),
                "pass": bool(semantic["pass"]),
                "judge_backend": semantic.get("judge_backend"),
                "judge_rubric_sha256": semantic.get("judge_rubric_sha256"),
            },
        }

    controlled = controlled_bundle_check(manifests)
    if not controlled:
        raise ValueError("C3 exposure matching or one-factor control audit failed")
    conclusion = transfer_conclusion(rows)
    return {
        "stage": "C3_exposure_matched_auxiliary_transfer_v1",
        "status": "complete",
        "diagnostic_only": True,
        "controlled_bundle_pass": controlled,
        "arms": rows,
        "conclusion": conclusion,
        "scientific_gate_pass": bool(conclusion["C3_transfer_hypothesis_supported"]),
        "next_action": (
            "replicate_recommended_auxiliary_target_with_matched_training_seeds"
            if conclusion["replication_recommended"]
            else "do_not_change_unified_recipe_from_C3"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--t3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.root, args.t3_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
