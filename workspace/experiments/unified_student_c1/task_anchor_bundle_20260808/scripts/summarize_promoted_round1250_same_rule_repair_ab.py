#!/usr/bin/env python3
"""Audit the controlled round-1250 same-rule repair A/B.

This diagnostic can recommend a multi-seed replication.  It deliberately
cannot promote a recipe from one training seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = {"base", "same_rule"}
T6_REFERENCE_MACRO_F1 = 0.5959866676584252
T6_MIN_RATIO = 0.85


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def weighted(groups: dict[str, Any], metric: str) -> float:
    rows = [row for name, row in groups.items() if name.startswith("predict_error_persistence/")]
    support = sum(int(row["n"]) for row in rows)
    if support != 976:
        raise ValueError(f"T6 support drift: {support}")
    return sum(float(row[metric]) * int(row["n"]) for row in rows) / support


def find_job(root: Path, variant: str) -> Path:
    matches: list[Path] = []
    for receipt_path in root.rglob("task_retention_training_receipt.json"):
        receipt = read(receipt_path)
        if receipt.get("causal_variant") != variant:
            continue
        for parent in receipt_path.parents:
            if (parent / "eval/c4/recurrent_causal_gate.json").is_file():
                matches.append(parent)
                break
            if parent == root:
                break
    if len(matches) != 1:
        raise ValueError(f"Expected one {variant} result, got {matches}")
    return matches[0]


def formal_scores(metrics: dict[str, Any]) -> dict[str, float]:
    groups = metrics["metrics_by_group"]
    task_metric = {
        "predict_correctness": "macro_f1",
        "predict_response": "math_equivalence",
        "predict_prospective_error": "joint_math_equivalence",
    }
    output: dict[str, float] = {}
    for task, metric in task_metric.items():
        rows = [row for name, row in groups.items() if name.startswith(task + "/")]
        support = sum(int(row["n"]) for row in rows)
        if support <= 0:
            raise ValueError(f"No formal support for {task}")
        output[task] = sum(float(row[metric]) * int(row["n"]) for row in rows) / support
    t3 = [row for name, row in groups.items() if name.startswith("diagnose_misconception/")]
    if sum(int(row["n"]) for row in t3) != 75:
        raise ValueError("T3 formal-screen support drift")
    output["T3_format_valid_rate"] = sum(
        float(row["format_valid_rate"]) * int(row["n"]) for row in t3
    ) / 75
    return output


def summarize(root: Path, t3_root: Path) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for variant in sorted(VARIANTS):
        job = find_job(root, variant)
        receipt = read(next(job.rglob("task_retention_training_receipt.json")))
        completion = read(next(job.rglob("metrics/completion.json")))
        config = read(next(job.rglob("metrics/run_config.json")))
        c4_payload = read(job / "eval/c4/recurrent_causal_gate.json")
        formal = read(job / "eval/formal_screen/metrics.json")
        t6 = read(job / "eval/t6_full/metrics.json")
        t3 = read(t3_root / variant / "gate.json")
        if t3.get("support") != 75 or t3.get("checkpoint_round") != 1500:
            raise ValueError(f"T3 semantic audit drift: {variant}")
        c4 = c4_payload["summary"]
        clean = c4["clean"]
        t6_macro = weighted(t6["metrics_by_group"], "macro_f1")
        t6_recall = weighted(t6["metrics_by_group"], "positive_recall")
        formal_summary = formal_scores(formal)
        checks = {
            "diagnostic_only": receipt.get("diagnostic_only") is True,
            "source_round_1250": receipt.get("source_checkpoint_round") == 1250,
            "train_support": completion.get("micro_steps") == 1000
            and completion.get("optimizer_steps") == 250,
            "causal_exposure": completion.get("causal_pairs_seen") == 50,
            "state_path_recipe": receipt.get("loss_profile") == "state_path"
            and float(config.get("causal_weight", -1)) == 2.0,
            "closed_state": c4_payload.get("state_architecture") == "layerwise"
            and c4_payload.get("raw_history_visible_to_decoder") is False
            and c4_payload.get("persistent_history_kv_cache") is False,
            "C4": c4.get("scientific_gate_pass") is True,
            "T3_semantic": t3.get("pass") is True,
            "formal_retention": (
                formal_summary["predict_correctness"] >= 0.40
                and formal_summary["predict_response"] >= 0.20
                and formal_summary["predict_prospective_error"] >= 0.40
                and formal_summary["T3_format_valid_rate"] >= 0.80
            ),
            "T6_retention": t6_macro / T6_REFERENCE_MACRO_F1 >= T6_MIN_RATIO
            and t6_recall > 0.0,
        }
        rows[variant] = {
            "causal_variant": variant,
            "checks": checks,
            "all_gates_pass": all(checks.values()),
            "C4": {
                "clean_n": int(c4["clean_n"]),
                "swap": float(clean["swap"]["flip_to_A_rate"]),
                "random": float(clean["random"]["flip_to_A_rate"]),
                "same_rule": float(clean["samemal"]["flip_to_A_rate"]),
                "noop": float(clean["noop"]["flip_to_A_rate"]),
                "D_specific": float(c4["D_specific"]),
                "scientific_gate_pass": bool(c4["scientific_gate_pass"]),
            },
            "formal_screen": formal_summary,
            "T3_semantic": {
                "support": int(t3["support"]),
                "semantic_accuracy": float(t3["semantic_accuracy"]),
                "retention_ratio": float(t3["retention_ratio"]),
                "pass": bool(t3["pass"]),
                "judge_backend": t3.get("judge_backend"),
                "judge_rubric_sha256": t3.get("judge_rubric_sha256"),
            },
            "T6": {
                "support": int(t6["sampled_rows"]),
                "macro_f1": t6_macro,
                "positive_recall": t6_recall,
                "retention_ratio": t6_macro / T6_REFERENCE_MACRO_F1,
            },
        }

    base = rows["base"]
    repaired = rows["same_rule"]
    control_improvement = base["C4"]["same_rule"] - repaired["C4"]["same_rule"]
    specificity_delta = repaired["C4"]["D_specific"] - base["C4"]["D_specific"]
    task_deltas = {
        key: repaired["formal_screen"][key] - base["formal_screen"][key]
        for key in repaired["formal_screen"]
    }
    replication_checks = {
        "same_rule_arm_passes_all_bounded_gates": repaired["all_gates_pass"],
        "same_rule_control_improves_by_0.05": control_improvement >= 0.05,
        "specificity_not_materially_worse": specificity_delta >= -0.03,
        "no_formal_task_drop_over_0.05": min(task_deltas.values()) >= -0.05,
        "T3_semantic_not_materially_worse": (
            repaired["T3_semantic"]["semantic_accuracy"]
            - base["T3_semantic"]["semantic_accuracy"]
        ) >= -0.05,
        "one_seed_only_not_promotable": True,
    }
    recommend = all(replication_checks.values())
    return {
        "stage": "promoted_round1250_same_rule_repair_ab",
        "diagnostic_only": True,
        "main_exact_epoch_unchanged": True,
        "rows": rows,
        "controlled_deltas_same_rule_minus_base": {
            "same_rule_control_improvement": control_improvement,
            "D_specific": specificity_delta,
            "formal_screen": task_deltas,
            "T6_macro_f1": repaired["T6"]["macro_f1"] - base["T6"]["macro_f1"],
            "T3_semantic_accuracy": (
                repaired["T3_semantic"]["semantic_accuracy"]
                - base["T3_semantic"]["semantic_accuracy"]
            ),
        },
        "replication_checks": replication_checks,
        "replication_recommended": recommend,
        "recipe_promotion_allowed": False,
        "next_action": (
            "run_matched_pair_seed_or_training_seed_replication"
            if recommend
            else "keep_frozen_main_recipe"
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
