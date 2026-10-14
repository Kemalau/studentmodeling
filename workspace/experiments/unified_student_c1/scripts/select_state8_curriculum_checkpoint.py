#!/usr/bin/env python3
"""Select a jointly qualified State8 curriculum checkpoint.

The selector is deliberately fail-closed: an average score cannot compensate
for a collapsed task, an exact-label shortcut, or either failed causal panel.
Only checkpoints that pass every frozen diagnostic floor are ranked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARMS = {
    "generic": "state8_diagnosis_first_eedi_curriculum_20260812",
    "current_query": "state8_diagnosis_first_eedi_current_query_curriculum_20260812",
}
STEPS = {2000, 3000, 4000, 5000}
FLOORS: dict[str, float] = {
    "T1_macro_f1": 0.55,
    "T2_math_equivalence": 0.20,
    "T3_semantic": 0.30,
    "T3_eedi_semantic": 0.20,
    "T3_synthetic_cross_semantic": 0.45,
    "T3_synthetic_same_semantic": 0.45,
    "T3_schema": 0.95,
    "T5_joint_math_equivalence": 0.35,
    "T6_macro_f1": 0.30,
    "T6_positive_recall": 0.0,
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def semantic_group(groups: dict[str, Any], fragment: str) -> float:
    rows = [row for name, row in groups.items() if fragment in name]
    if len(rows) != 1:
        raise ValueError(f"Expected one semantic group for {fragment}")
    if int(rows[0].get("llm_judge_n", -1)) != 25:
        raise ValueError(f"Semantic support drift for {fragment}")
    return float(rows[0]["llm_judge_accuracy"])


def diagnosis_checks(prefix: str, value: dict[str, Any]) -> dict[str, Any]:
    if value.get("semantic_judge_authoritative") is not True:
        raise ValueError(f"{prefix} semantic judge is not authoritative")
    if value.get("exact_label_role") != "auxiliary_only_not_used_for_gate":
        raise ValueError(f"{prefix} exact-label role drift")
    cells = value.get("cells") or {}
    generation = value.get("generation") or {}
    return {
        f"{prefix}_pass": value.get("pass") is True,
        f"{prefix}_D_specific": float(value.get("semantic_D_specific", -1)),
        f"{prefix}_swap_to_donor": float(
            (cells.get("swap_to_donor") or {}).get("semantic_equivalence_rate", -1)
        ),
        f"{prefix}_random_stable": float(
            value.get("random_matches_either_stable_target_rate", 1)
        ),
        f"{prefix}_noop_retention": float(
            value.get("noop_recipient_retention", -1)
        ),
        f"{prefix}_same_rule_retention": float(
            value.get("same_rule_recipient_retention", -1)
        ),
        f"{prefix}_artifact_sha256": generation.get("artifact_sha256"),
    }


def evaluate(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("stage") != "state8_curriculum_checkpoint_unified_semantic_summary_v1":
        raise ValueError("Curriculum summary stage drift")
    if summary.get("status") != "complete" or summary.get("diagnostic_only") is not True:
        raise ValueError("Curriculum checkpoint screen is incomplete")
    if summary.get("semantic_judge_authoritative") is not True:
        raise ValueError("T3 semantic judge is not authoritative")
    if summary.get("exact_match_authoritative") is not False:
        raise ValueError("Exact match became authoritative")
    arm = str(summary.get("arm") or "")
    step = int(summary.get("checkpoint_step", -1))
    if arm not in ARMS or step not in STEPS:
        raise ValueError(f"Curriculum coordinate drift: {arm}/{step}")
    if summary.get("state_shape") != [64, 8, 5120]:
        raise ValueError("State shape drift")
    if summary.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Raw history leaked to decoder")
    artifacts = summary.get("artifact_sha256") or {}
    if set(artifacts) != {"adapter_model", "recurrent_state"} or any(
        not isinstance(value, str) or len(value) != 64 for value in artifacts.values()
    ):
        raise ValueError("Checkpoint artifact binding drift")

    tasks = summary.get("task_metrics") or {}
    t3 = summary.get("T3_semantic") or {}
    groups = t3.get("metrics_by_group") or {}
    if int(t3.get("support", -1)) != 75:
        raise ValueError("T3 screen support drift")
    malrule = diagnosis_checks(
        "malrule_diagnosis_C4", summary.get("malrule_diagnosis_C4") or {}
    )
    eedi = diagnosis_checks(
        "eedi_same_question_C4", summary.get("eedi_same_question_C4") or {}
    )
    for name, value in (
        ("malrule", malrule["malrule_diagnosis_C4_artifact_sha256"]),
        ("eedi", eedi["eedi_same_question_C4_artifact_sha256"]),
    ):
        if value != artifacts:
            raise ValueError(f"{name} C4 artifact binding drift")

    checks: dict[str, Any] = {
        "T1_macro_f1": float(tasks["predict_correctness"]["score"]),
        "T2_math_equivalence": float(tasks["predict_response"]["score"]),
        "T3_semantic": float(t3["accuracy"]),
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
        ) / sum(int(row["llm_judge_n"]) for row in groups.values()),
        "T5_joint_math_equivalence": float(
            tasks["predict_prospective_error"]["score"]
        ),
        "T6_macro_f1": float(tasks["predict_error_persistence"]["score"]),
        "T6_positive_recall": float(summary["T6_positive_recall"]["score"]),
        "answer_C4_pass": (summary.get("answer_C4") or {}).get("pass") is True,
        "answer_C4_D_specific": float(
            (summary.get("answer_C4") or {}).get("D_specific", -1)
        ),
        **{key: value for key, value in malrule.items() if not key.endswith("artifact_sha256")},
        **{key: value for key, value in eedi.items() if not key.endswith("artifact_sha256")},
    }
    failed = [
        name
        for name, floor in FLOORS.items()
        if (
            checks[name] <= floor
            if name == "T6_positive_recall"
            else checks[name] < floor
        )
    ]
    for name in (
        "answer_C4_pass",
        "malrule_diagnosis_C4_pass",
        "eedi_same_question_C4_pass",
    ):
        if checks[name] is not True:
            failed.append(name)
    score = (
        0.10 * checks["T1_macro_f1"]
        + 0.08 * checks["T2_math_equivalence"]
        + 0.12 * checks["T3_semantic"]
        + 0.15 * checks["T3_eedi_semantic"]
        + 0.05 * checks["T3_synthetic_cross_semantic"]
        + 0.05 * checks["T3_synthetic_same_semantic"]
        + 0.08 * checks["T5_joint_math_equivalence"]
        + 0.08 * checks["T6_macro_f1"]
        + 0.08 * max(0.0, checks["answer_C4_D_specific"])
        + 0.08 * max(0.0, checks["malrule_diagnosis_C4_D_specific"])
        + 0.10 * max(0.0, checks["eedi_same_question_C4_D_specific"])
        - 0.04 * checks["malrule_diagnosis_C4_random_stable"]
        - 0.06 * checks["eedi_same_question_C4_random_stable"]
    )
    slug = f"curriculum_{arm}_cp{step}"
    return {
        "arm": slug,
        "curriculum_arm": arm,
        "checkpoint_step": step,
        "slug": slug,
        "qualified": not failed,
        "failed_checks": failed,
        "score": score,
        "checks": checks,
        "floors": FLOORS,
        "artifact_sha256": artifacts,
        "remote_root": (
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            f"{ARMS[arm]}/checkpoint-{step}"
        ),
        "finalfit_variant": "same_question_curriculum_promoted",
        "source_history_mode": "five_stage_curriculum",
    }


def select(values: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [evaluate(value) for value in values]
    coordinates = {
        (row["curriculum_arm"], row["checkpoint_step"]) for row in candidates
    }
    expected = {(arm, step) for arm in ARMS for step in STEPS}
    if coordinates != expected:
        raise ValueError(f"Curriculum screen matrix drift: {sorted(coordinates)}")
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(
        eligible,
        key=lambda row: (row["score"], row["checkpoint_step"], row["curriculum_arm"]),
    ) if eligible else None
    return {
        "stage": "state8_curriculum_checkpoint_joint_selection_v1",
        "status": "complete",
        "pass": selected is not None,
        "selected": selected,
        "selected_arm": selected["arm"] if selected else None,
        "candidates": sorted(
            candidates, key=lambda row: (row["curriculum_arm"], row["checkpoint_step"])
        ),
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "state_token_count": 8,
        "raw_history_visible_to_decoder": False,
        "answer_C4_confirmation_pending": selected is not None,
        "full_exact_epoch_pending": selected is not None,
        "selection_rule": "highest_score_after_all_hard_task_and_causal_gates",
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
