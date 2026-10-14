#!/usr/bin/env python3
"""Summarize one bounded task-retention candidate with frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_layerwise_checkpoint250_formal_screen import summarize as summarize_formal
from validate_checkpoint_t3_semantic_result import validate as validate_t3


T6_REFERENCE_MACRO_F1 = 0.5959866676584252
T6_MIN_RATIO = 0.85


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def weighted(groups: dict[str, Any], metric: str) -> float:
    support = sum(int(row["n"]) for row in groups.values())
    if support <= 0:
        raise ValueError("Metric support must be positive")
    return sum(float(row[metric]) * int(row["n"]) for row in groups.values()) / support


def candidate_root(
    root: Path,
    branch: str,
    loss_profile: str,
    learning_rate: float | None = None,
    causal_variant: str | None = None,
) -> Path:
    receipts = []
    for path in root.rglob("task_retention_training_receipt.json"):
        value = read(path)
        observed_lr = value.get("learning_rate")
        lr_matches = learning_rate is None or (
            observed_lr is not None and abs(float(observed_lr) - learning_rate) < 1e-15
        )
        variant_matches = causal_variant is None or value.get("causal_variant") == causal_variant
        if (
            value.get("branch") == branch
            and value.get("loss_profile") == loss_profile
            and lr_matches
            and variant_matches
        ):
            receipts.append(path)
    if len(receipts) != 1:
        raise ValueError(
            f"Expected one receipt for branch={branch} profile={loss_profile}: {receipts}"
        )
    for ancestor in receipts[0].parents:
        if (ancestor / "eval/formal_screen/metrics.json").is_file():
            return ancestor
        if ancestor == root:
            break
    raise ValueError("Could not locate candidate evaluation root")


def validate_open_t3_training_view(
    job: Path, receipt: dict[str, Any], loss_profile: str
) -> bool:
    """Validate the audited train-only prompt/KD view for open-T3 candidates."""

    if loss_profile == "state_path":
        paths = list(job.rglob("open_t3_training_view.json"))
        expected_stage = "open_t3_training_view_v1"
        expected_training_view = "open_t3_no_candidate_enumeration"
        extra_checks = lambda value: (
            float(value.get("behavior_kd_weight", -1.0)) == 0.0
            and float(value.get("direct_replay_weight", -1.0)) == 0.0
        )
    elif loss_profile == "state_kd":
        paths = list(job.rglob("open_t3_selective_kd_training_view.json"))
        expected_stage = "open_t3_selective_kd_training_view_v1"
        expected_training_view = "open_t3_selective_kd"
        extra_checks = lambda value: (
            float(value.get("global_behavior_kd_weight", -1.0)) > 0.0
            and float(value.get("excluded_behavior_kd_weight", -1.0)) == 0.0
            and float(value.get("direct_replay_weight", -1.0)) == 0.0
            and int(value.get("excluded_history_rows", 0)) > 0
            and int(value.get("retained_history_rows", 0)) > 0
        )
    else:
        return False
    if len(paths) != 1:
        return False
    value = read(paths[0])
    return bool(
        value.get("stage") == expected_stage
        and value.get("scope") == "train_jsonl_only"
        and int(value.get("changed_malrule_t3_rows", 0)) > 0
        and value.get("causal_rows_modified") is False
        and value.get("validation_rows_modified") is False
        and receipt.get("training_view") == expected_training_view
        and extra_checks(value)
    )


def summarize(
    root: Path,
    t3_metrics: Path,
    *,
    branch: str,
    loss_profile: str,
    learning_rate: float | None = None,
    checkpoint_round: int = 750,
    causal_variant: str | None = None,
) -> dict[str, Any]:
    job = candidate_root(root, branch, loss_profile, learning_rate, causal_variant)
    receipt = read(next(job.rglob("task_retention_training_receipt.json")))
    completion = read(next(job.rglob("metrics/completion.json")))
    run_config = read(next(job.rglob("metrics/run_config.json")))
    c4 = read(job / "eval/c4/recurrent_causal_gate.json")
    formal = summarize_formal(
        job / "eval/formal_screen", checkpoint_round=checkpoint_round
    )
    t6 = read(job / "eval/t6_full/metrics.json")
    t3 = validate_t3(t3_metrics, checkpoint_round=checkpoint_round)
    seen_unseen_path = t3_metrics.parent / "seen_unseen.json"
    t3_seen_unseen = read(seen_unseen_path) if seen_unseen_path.is_file() else None
    if t3_seen_unseen is not None:
        if (
            t3_seen_unseen.get("stage")
            != "eedi_t3_seen_unseen_taxonomy_diagnostic_v1"
            or t3_seen_unseen.get("diagnostic_only") is not True
        ):
            raise ValueError("Eedi T3 seen/unseen diagnostic drift")
    c4_summary = c4["summary"]
    clean = c4_summary["clean"]
    t6_groups = t6["metrics_by_group"]
    t6_macro = weighted(t6_groups, "macro_f1")
    t6_recall = weighted(t6_groups, "positive_recall")
    checks = {
        "diagnostic_only": receipt.get("diagnostic_only") is True
        and receipt.get("exact_resume") is False,
        "matched_candidate": receipt.get("branch") == branch
        and receipt.get("loss_profile") == loss_profile,
        "train_support": completion.get("micro_steps") == receipt.get("micro_steps")
        and completion.get("optimizer_steps") == receipt.get("optimizer_steps"),
        "train_boundary": completion.get("history_global_position_start")
        == receipt.get("start_global_position")
        and completion.get("history_global_position_end")
        == receipt.get("end_global_position"),
        "causal_exposure": completion.get("causal_pairs_seen")
        == receipt.get("causal_pairs_seen"),
        "causal_recipe": float(run_config.get("causal_weight", -1.0)) == 2.0
        and float(receipt.get("causal_weight", run_config.get("causal_weight", -1.0)))
        == 2.0,
        "formal_test_overlap_zero": receipt.get("formal_example_id_overlap") == 0,
        "closed_layerwise": c4.get("state_architecture") == "layerwise"
        and c4.get("raw_history_visible_to_decoder") is False
        and c4.get("persistent_history_kv_cache") is False,
        "C4": bool(c4_summary.get("scientific_gate_pass")),
        "formal_screen": bool(formal.get("diagnostic_gate_pass")),
        "T3_semantic": bool(t3.get("pass")),
        "T6_retention": t6_macro / T6_REFERENCE_MACRO_F1 >= T6_MIN_RATIO
        and t6_recall > 0.0,
    }
    if branch in {"eedi_full_open_t3", "equal_t3_t6_open_t3"}:
        checks["open_t3_training_view"] = validate_open_t3_training_view(
            job, receipt, loss_profile
        )
    return {
        "stage": "layerwise_single_task_retention_candidate",
        "candidate_id": (
            f"{branch}__{loss_profile}"
            if learning_rate is None
            else f"{branch}__{loss_profile}__lr{learning_rate:.0e}"
        ),
        "branch": branch,
        "loss_profile": loss_profile,
        "learning_rate": float(receipt.get("learning_rate", learning_rate or 5e-7)),
        "causal_weight": float(run_config["causal_weight"]),
        "causal_variant": receipt.get("causal_variant", "base"),
        "diagnostic_only": True,
        "checks": checks,
        "all_gates_pass": all(checks.values()),
        "task_counts": receipt["task_counts"],
        "C4": {
            "clean_n": c4_summary["clean_n"],
            "swap": clean["swap"]["flip_to_A_rate"],
            "random": clean["random"]["flip_to_A_rate"],
            "same_rule": clean["samemal"]["flip_to_A_rate"],
            "noop": clean["noop"]["flip_to_A_rate"],
            "D_specific": c4_summary["D_specific"],
        },
        "formal_screen": formal,
        "T3_semantic": t3,
        "T3_seen_unseen": t3_seen_unseen,
        "T6": {
            "support": int(t6["sampled_rows"]),
            "macro_f1": t6_macro,
            "reference_macro_f1": T6_REFERENCE_MACRO_F1,
            "retention_ratio": t6_macro / T6_REFERENCE_MACRO_F1,
            "positive_recall": t6_recall,
            "by_group": t6_groups,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--t3-metrics", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument(
        "--loss-profile",
        choices=("standard", "state_kd", "state_path"),
        required=True,
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoint-round", type=int, default=750)
    parser.add_argument("--causal-variant", choices=("base", "same_rule"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(
        args.root,
        args.t3_metrics,
        branch=args.branch,
        loss_profile=args.loss_profile,
        learning_rate=args.learning_rate,
        checkpoint_round=args.checkpoint_round,
        causal_variant=args.causal_variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
