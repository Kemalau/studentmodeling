#!/usr/bin/env python3
"""Combine the immutable round-1000 C4 and task diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def summarize(
    c4: dict[str, Any], formal: dict[str, Any], t3_semantic: dict[str, Any]
) -> dict[str, Any]:
    if c4.get("stage") != "layerwise_finalfit_checkpoint1000_c4":
        raise ValueError("C4 stage drift")
    if c4.get("diagnostic_only") is not True or c4.get("formal_final_c4_required") is not True:
        raise ValueError("C4 result is not the diagnostic-only frozen checkpoint")
    if c4.get("exact_resume_ready") is not True:
        raise ValueError("Round-1000 checkpoint is not proven exact-resume ready")
    if formal.get("stage") != "layerwise_finalfit_checkpoint1000_formal_screen":
        raise ValueError("Formal-screen stage drift")
    if formal.get("diagnostic_only") is not True or formal.get("formal_final_evaluation_required") is not True:
        raise ValueError("Formal result is not the diagnostic-only frozen checkpoint")
    if t3_semantic.get("stage") != "layerwise_finalfit_checkpoint1000_T3_semantic_judge":
        raise ValueError("T3 semantic stage drift")
    if t3_semantic.get("diagnostic_only") is not True:
        raise ValueError("T3 semantic result is not diagnostic-only")
    if int(t3_semantic.get("support", 0)) != 75:
        raise ValueError("T3 semantic result does not cover the frozen 75-row panel")

    c4_pass = bool(c4.get("scientific_gate_pass_at_checkpoint"))
    t3_semantic_pass = bool(t3_semantic.get("pass"))
    formal_pass = bool(formal.get("diagnostic_gate_pass")) and t3_semantic_pass
    if c4_pass and formal_pass:
        diagnosis = "healthy_early_checkpoint"
        next_action = "continue_exact_epoch_and_run_mandatory_final_evaluation"
    elif not c4_pass and formal_pass:
        diagnosis = "causal_specificity_at_risk"
        next_action = "preserve_main_fit_and_prepare_causal_only_checkpoint_rescue"
    elif c4_pass and not formal_pass:
        diagnosis = "task_retention_at_risk"
        next_action = "preserve_main_fit_and_prepare_task_retention_checkpoint_rescue"
    else:
        diagnosis = "joint_training_instability_at_risk"
        next_action = "preserve_main_fit_and_prepare_joint_checkpoint_rescue"

    return {
        "stage": "layerwise_unified_round1000_joint_decision",
        "checkpoint_round": 1000,
        "diagnostic_only": True,
        "main_fit_must_continue": True,
        "formal_final_evaluation_required": True,
        "C4": {
            "pass": c4_pass,
            "exact_resume_ready": True,
            "clean_n": int(c4["clean_n"]),
            "swap": float(c4["swap"]),
            "random": float(c4["random"]),
            "same_rule": float(c4["same_rule"]),
            "noop": float(c4["noop"]),
            "D_specific": float(c4["D_specific"]),
        },
        "formal_screen": {
            "pass": formal_pass,
            "sampled_rows": int(formal["sampled_rows"]),
            "performance": formal["performance"],
            "T3_format_valid_rate": float(formal["T3_format_valid_rate"]),
        },
        "T3_semantic": {
            "pass": t3_semantic_pass,
            "support": int(t3_semantic["support"]),
            "semantic_accuracy": float(t3_semantic["semantic_accuracy"]),
            "retention_ratio": float(t3_semantic["retention_ratio"]),
            "minimum_retention_ratio": float(t3_semantic["minimum_retention_ratio"]),
            "by_group": t3_semantic["by_group"],
        },
        "diagnosis": diagnosis,
        "next_action": next_action,
        "rescue_preparation_required": not (c4_pass and formal_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c4", type=Path, required=True)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--t3-semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(read(args.c4), read(args.formal), read(args.t3_semantic))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
