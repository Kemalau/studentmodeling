#!/usr/bin/env python3
"""Select the best causal-teacher repair checkpoint under the shared gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from select_state8_curriculum_repair_checkpoint import (
        diagnosis_checks,
        read,
        semantic_group,
    )
    from select_state8_global_candidate import FLOORS, REPAIR_STAGE
except ImportError:  # package-style unit tests
    from scripts.select_state8_curriculum_repair_checkpoint import (
        diagnosis_checks,
        read,
        semantic_group,
    )
    from scripts.select_state8_global_candidate import FLOORS, REPAIR_STAGE


SPECS = {
    (8201, "causal_teacher_low"),
    (8202, "causal_teacher_high"),
}
STEPS = {20, 100, 240, 500}


def evaluate(
    summary: dict[str, Any],
    specs: set[tuple[int, str]] = SPECS,
) -> dict[str, Any]:
    if summary.get("stage") != "state8_curriculum_checkpoint_unified_semantic_summary_v1":
        raise ValueError("Causal-teacher checkpoint summary stage drift")
    if summary.get("status") != "complete" or summary.get("diagnostic_only") is not True:
        raise ValueError("Causal-teacher checkpoint screen is incomplete")
    if summary.get("semantic_judge_authoritative") is not True:
        raise ValueError("Semantic judge is not authoritative")
    if summary.get("exact_match_authoritative") is not False:
        raise ValueError("Exact match became authoritative")
    if summary.get("state_shape") != [64, 8, 5120]:
        raise ValueError("State shape drift")
    if summary.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Raw history leaked to decoder")

    arm = str(summary.get("arm") or "")
    match = re.fullmatch(
        r"repair_(causal_teacher_low|causal_teacher_high|early_label_low|early_label_high)_seed(\d+)",
        arm,
    )
    if match is None:
        raise ValueError(f"Causal-teacher arm drift: {arm}")
    branch, seed_text = match.groups()
    seed = int(seed_text)
    step = int(summary.get("checkpoint_step", -1))
    if (seed, branch) not in specs or step not in STEPS:
        raise ValueError(f"Causal-teacher coordinate drift: {seed}/{branch}/{step}")

    artifacts = summary.get("artifact_sha256") or {}
    if set(artifacts) != {"adapter_model", "recurrent_state"} or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in artifacts.values()
    ):
        raise ValueError("Causal-teacher checkpoint artifact binding drift")
    tasks = summary.get("task_metrics") or {}
    t3 = summary.get("T3_semantic") or {}
    groups = t3.get("metrics_by_group") or {}
    if int(t3.get("support", -1)) != 75:
        raise ValueError("T3 support drift")
    schema_denominator = sum(int(row["llm_judge_n"]) for row in groups.values())
    if schema_denominator != 75:
        raise ValueError("T3 schema support drift")

    malrule = summary.get("malrule_diagnosis_C4") or {}
    eedi = summary.get("eedi_same_question_C4") or {}
    for name, value in (("malrule", malrule), ("eedi", eedi)):
        if (value.get("generation") or {}).get("artifact_sha256") != artifacts:
            raise ValueError(f"{name} C4 artifact binding drift")

    checks: dict[str, Any] = {
        "T1": float(tasks["predict_correctness"]["score"]),
        "T2": float(tasks["predict_response"]["score"]),
        "T3_eedi_semantic": float(t3["eedi_accuracy"]),
        "T3_synthetic_cross_semantic": semantic_group(
            groups, "malrule/synthetic_cross_template"
        ),
        "T3_synthetic_same_semantic": semantic_group(
            groups, "malrule/synthetic_same_template"
        ),
        "T3_schema": sum(
            float(row["natural_schema_valid_rate"]) * int(row["llm_judge_n"])
            for row in groups.values()
        ) / schema_denominator,
        "T5": float(tasks["predict_prospective_error"]["score"]),
        "T6": float(tasks["predict_error_persistence"]["score"]),
        "T6_macro_f1": float(tasks["predict_error_persistence"]["score"]),
        "T6_positive_recall": float(summary["T6_positive_recall"]["score"]),
        **diagnosis_checks("malrule_diagnosis_C4", malrule),
        **diagnosis_checks("eedi_same_question_C4", eedi),
    }
    projected = {
        "T1_macro_f1": checks["T1"],
        "T2_math_equivalence": checks["T2"],
        "T3_eedi_semantic": checks["T3_eedi_semantic"],
        "T3_synthetic_cross_semantic": checks["T3_synthetic_cross_semantic"],
        "T3_synthetic_same_semantic": checks["T3_synthetic_same_semantic"],
        "T3_schema": checks["T3_schema"],
        # Public taxonomy v2 renumbered legacy T5 prospective behavior to T4
        # and legacy T6 persistence to T5.  ``checks`` keeps the legacy names
        # for artifact compatibility, while FLOORS is intentionally public.
        "T4_joint_math_equivalence": checks["T5"],
        "T5_macro_f1": checks["T6_macro_f1"],
        "T5_positive_recall": checks["T6_positive_recall"],
    }
    failed = [
        name
        for name, floor in FLOORS.items()
        if (
            float(projected[name]) <= floor
            if name == "T5_positive_recall"
            else float(projected[name]) < floor
        )
    ]
    for name in ("malrule_diagnosis_C4_pass", "eedi_same_question_C4_pass"):
        if checks[name] is not True:
            failed.append(name)
    score = (
        0.10 * checks["T1"]
        + 0.08 * checks["T2"]
        + 0.16 * checks["T3_eedi_semantic"]
        + 0.05 * checks["T3_synthetic_cross_semantic"]
        + 0.05 * checks["T3_synthetic_same_semantic"]
        + 0.08 * checks["T5"]
        + 0.08 * checks["T6_macro_f1"]
        + 0.08 * max(0.0, checks["malrule_diagnosis_C4_D_specific"])
        + 0.11 * max(0.0, checks["eedi_same_question_C4_D_specific"])
        + 0.04 * checks["malrule_diagnosis_C4_swap_to_donor"]
        + 0.06 * checks["eedi_same_question_C4_swap_to_donor"]
        + 0.02 * checks["eedi_same_question_C4_noop_retention"]
        + 0.02 * checks["eedi_same_question_C4_same_rule_retention"]
        - 0.03 * checks["malrule_diagnosis_C4_random_stable"]
        - 0.05 * checks["eedi_same_question_C4_random_stable"]
    )
    return {
        "arm": f"repair_checkpoint_{branch}_seed{seed}_cp{step}",
        "qualified": not failed,
        "failed_checks": failed,
        "score": score,
        "checks": checks,
        "floors": FLOORS,
        "artifact_sha256": artifacts,
        "remote_root": (
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            f"state8_curriculum_repairs_20260812/seed{seed}_{branch}/checkpoint-{step}"
        ),
        "artifact_layout": "repair_checkpoint_snapshot",
        "finalfit_variant": "same_question_repair_checkpoint_promoted",
        "repair_seed": seed,
        "repair_branch": branch,
        "checkpoint_step": step,
        "source_checkpoint_step": 3000,
        "source_history_mode": (
            "five_stage_curriculum_then_strict_state_only_early_label_binding"
            if branch.startswith("early_label_")
            else "five_stage_curriculum_then_strict_state_only_causal_teacher_binding"
        ),
    }


def select(values: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [evaluate(value) for value in values]
    observed = {
        (row["repair_seed"], row["repair_branch"], row["checkpoint_step"])
        for row in candidates
    }
    expected = {
        (seed, branch, step)
        for seed, branch in SPECS
        for step in STEPS
    }
    if observed != expected:
        raise ValueError(f"Causal-teacher checkpoint matrix drift: {sorted(observed)}")
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(eligible, key=lambda row: (row["score"], row["arm"])) \
        if eligible else None
    return {
        "stage": REPAIR_STAGE,
        "status": "complete",
        "pass": selected is not None,
        "selected": selected,
        "selected_arm": selected["arm"] if selected else None,
        "candidates": sorted(candidates, key=lambda row: row["arm"]),
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "raw_history_visible_to_decoder": False,
        "state_token_count": 8,
        "answer_C4_confirmation_pending": selected is not None,
        "full_exact_epoch_pending": selected is not None,
        "selection_rule": (
            "highest_shared_score_after_all_task_and_dual_c4_gates_across_"
            "causal_teacher_low_high_checkpoint20_100_240_and_500"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select([read(path) for path in args.candidate])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
