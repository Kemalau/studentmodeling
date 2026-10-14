#!/usr/bin/env python3
"""Select one State8 model across curriculum and state-only repair lineages.

Source selectors have different historical score definitions.  This module
does not compare those scores.  It projects every already-qualified source
candidate onto one shared set of task, semantic, schema, and causal metrics,
re-applies the common hard gates, and then computes one common score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from task_taxonomy_v2 import taxonomy_receipt
except ImportError:  # package-style tests
    from scripts.task_taxonomy_v2 import taxonomy_receipt


CURRICULUM_STAGE = "state8_curriculum_checkpoint_joint_selection_v1"
REPAIR_STAGE = "state8_eedi_same_question_joint_diagnostic_selection_v1"

FLOORS: dict[str, float] = {
    "T1_macro_f1": 0.55,
    "T2_math_equivalence": 0.20,
    "T3_eedi_semantic": 0.20,
    "T3_synthetic_cross_semantic": 0.45,
    "T3_synthetic_same_semantic": 0.45,
    "T3_schema": 0.95,
    "T4_joint_math_equivalence": 0.35,
    "T5_macro_f1": 0.30,
    "T5_positive_recall": 0.0,
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def common_checks(selection: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = str(selection.get("stage") or "")
    selected = selection.get("selected") or {}
    if selection.get("status") != "complete":
        raise ValueError("Source selection is incomplete")
    if selection.get("pass") is not True:
        raise ValueError("Source selection has no qualified candidate")
    if selected.get("qualified") is not True:
        raise ValueError("Source selected candidate is not qualified")
    checks = selected.get("checks") or {}
    if stage == CURRICULUM_STAGE:
        projected = {
            "T1_macro_f1": checks["T1_macro_f1"],
            "T2_math_equivalence": checks["T2_math_equivalence"],
            "T3_eedi_semantic": checks["T3_eedi_semantic"],
            "T3_synthetic_cross_semantic": checks[
                "T3_synthetic_cross_semantic"
            ],
            "T3_synthetic_same_semantic": checks[
                "T3_synthetic_same_semantic"
            ],
            "T3_schema": checks["T3_schema"],
            "T4_joint_math_equivalence": checks["T5_joint_math_equivalence"],
            "T5_macro_f1": checks["T6_macro_f1"],
            "T5_positive_recall": checks["T6_positive_recall"],
            "malrule_diagnosis_C4_pass": checks[
                "malrule_diagnosis_C4_pass"
            ],
            "malrule_diagnosis_C4_D_specific": checks[
                "malrule_diagnosis_C4_D_specific"
            ],
            "malrule_diagnosis_C4_swap_to_donor": checks[
                "malrule_diagnosis_C4_swap_to_donor"
            ],
            "malrule_diagnosis_C4_random_stable": checks[
                "malrule_diagnosis_C4_random_stable"
            ],
            "malrule_diagnosis_C4_noop_retention": checks[
                "malrule_diagnosis_C4_noop_retention"
            ],
            "malrule_diagnosis_C4_same_rule_retention": checks[
                "malrule_diagnosis_C4_same_rule_retention"
            ],
            "eedi_same_question_C4_pass": checks[
                "eedi_same_question_C4_pass"
            ],
            "eedi_same_question_C4_D_specific": checks[
                "eedi_same_question_C4_D_specific"
            ],
            "eedi_same_question_C4_swap_to_donor": checks[
                "eedi_same_question_C4_swap_to_donor"
            ],
            "eedi_same_question_C4_random_stable": checks[
                "eedi_same_question_C4_random_stable"
            ],
            "eedi_same_question_C4_noop_retention": checks[
                "eedi_same_question_C4_noop_retention"
            ],
            "eedi_same_question_C4_same_rule_retention": checks[
                "eedi_same_question_C4_same_rule_retention"
            ],
        }
    elif stage == REPAIR_STAGE:
        projected = {
            "T1_macro_f1": checks["T1"],
            "T2_math_equivalence": checks["T2"],
            "T3_eedi_semantic": checks["T3_eedi_semantic"],
            "T3_synthetic_cross_semantic": checks[
                "T3_synthetic_cross_semantic"
            ],
            "T3_synthetic_same_semantic": checks[
                "T3_synthetic_same_semantic"
            ],
            "T3_schema": checks["T3_schema"],
            "T4_joint_math_equivalence": checks["T5"],
            "T5_macro_f1": checks["T6_macro_f1"],
            "T5_positive_recall": checks["T6_positive_recall"],
            "malrule_diagnosis_C4_pass": checks[
                "malrule_diagnosis_C4_pass"
            ],
            "malrule_diagnosis_C4_D_specific": checks[
                "malrule_diagnosis_C4_D_specific"
            ],
            "malrule_diagnosis_C4_swap_to_donor": checks[
                "malrule_diagnosis_C4_swap_to_donor"
            ],
            "malrule_diagnosis_C4_random_stable": checks[
                "malrule_diagnosis_C4_random_stable"
            ],
            "malrule_diagnosis_C4_noop_retention": checks[
                "malrule_diagnosis_C4_noop_retention"
            ],
            "malrule_diagnosis_C4_same_rule_retention": checks[
                "malrule_diagnosis_C4_same_rule_retention"
            ],
            "eedi_same_question_C4_pass": checks[
                "eedi_same_question_C4_pass"
            ],
            "eedi_same_question_C4_D_specific": checks[
                "eedi_same_question_C4_D_specific"
            ],
            "eedi_same_question_C4_swap_to_donor": checks[
                "eedi_same_question_C4_swap_to_donor"
            ],
            "eedi_same_question_C4_random_stable": checks[
                "eedi_same_question_C4_random_stable"
            ],
            "eedi_same_question_C4_noop_retention": checks[
                "eedi_same_question_C4_noop_retention"
            ],
            "eedi_same_question_C4_same_rule_retention": checks[
                "eedi_same_question_C4_same_rule_retention"
            ],
        }
    else:
        raise ValueError(f"Unsupported source selection stage: {stage}")
    return selected, projected


def evaluate(source: str, selection: dict[str, Any]) -> dict[str, Any]:
    selected, checks = common_checks(selection)
    artifacts = selected.get("artifact_sha256") or {}
    if set(artifacts) != {"adapter_model", "recurrent_state"} or any(
        not isinstance(value, str) or len(value) != 64
        for value in artifacts.values()
    ):
        raise ValueError("Global candidate artifact binding drift")
    arm = str(selected.get("arm") or selection.get("selected_arm") or "")
    if not (arm.startswith("curriculum_") or arm.startswith("repair_")):
        raise ValueError(f"Global candidate arm drift: {arm}")
    remote_root = str(selected.get("remote_root") or "")
    if not remote_root.startswith("/mnt/amulet/"):
        raise ValueError("Global candidate remote root drift")
    layout = selected.get("artifact_layout")
    if layout is not None:
        if layout != "repair_checkpoint_snapshot":
            raise ValueError("Global candidate artifact layout drift")
        if not arm.startswith("repair_checkpoint_"):
            raise ValueError("Repair checkpoint arm/layout mismatch")
        if not remote_root.startswith(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_curriculum_repairs_20260812/"
        ):
            raise ValueError("Repair checkpoint root/layout mismatch")
    failed = [
        name
        for name, floor in FLOORS.items()
        if (
            float(checks[name]) <= floor
            if name == "T5_positive_recall"
            else float(checks[name]) < floor
        )
    ]
    for name in (
        "malrule_diagnosis_C4_pass",
        "eedi_same_question_C4_pass",
    ):
        if checks[name] is not True:
            failed.append(name)
    score = (
        0.10 * float(checks["T1_macro_f1"])
        + 0.08 * float(checks["T2_math_equivalence"])
        + 0.16 * float(checks["T3_eedi_semantic"])
        + 0.05 * float(checks["T3_synthetic_cross_semantic"])
        + 0.05 * float(checks["T3_synthetic_same_semantic"])
        + 0.08 * float(checks["T4_joint_math_equivalence"])
        + 0.08 * float(checks["T5_macro_f1"])
        + 0.08 * max(0.0, float(checks["malrule_diagnosis_C4_D_specific"]))
        + 0.11 * max(0.0, float(checks["eedi_same_question_C4_D_specific"]))
        + 0.04 * float(checks["malrule_diagnosis_C4_swap_to_donor"])
        + 0.06 * float(checks["eedi_same_question_C4_swap_to_donor"])
        + 0.02 * float(checks["eedi_same_question_C4_noop_retention"])
        + 0.02 * float(checks["eedi_same_question_C4_same_rule_retention"])
        - 0.03 * float(checks["malrule_diagnosis_C4_random_stable"])
        - 0.05 * float(checks["eedi_same_question_C4_random_stable"])
    )
    return {
        **selected,
        "arm": arm,
        "source_selection": source,
        "source_stage": selection["stage"],
        "source_score_diagnostic_only": selected.get("score"),
        "score": score,
        "checks": checks,
        "floors": FLOORS,
        "failed_checks": failed,
        "qualified": not failed,
        "task_taxonomy": taxonomy_receipt(),
    }


def select(values: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    if not values:
        raise ValueError("Global selection requires at least one source")
    sources = [source for source, _ in values]
    if len(sources) != len(set(sources)):
        raise ValueError("Duplicate global selection source")
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source, selection in values:
        if selection.get("status") != "complete":
            raise ValueError(f"Incomplete global selection source: {source}")
        if selection.get("pass") is True:
            candidates.append(evaluate(source, selection))
        else:
            skipped.append(
                {
                    "source": source,
                    "stage": selection.get("stage"),
                    "reason": "no_source_qualified_candidate",
                }
            )
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(eligible, key=lambda row: (row["score"], row["arm"])) \
        if eligible else None
    return {
        "stage": "state8_global_joint_diagnostic_selection_v1",
        "status": "complete",
        "task_taxonomy": taxonomy_receipt(),
        "pass": selected is not None,
        "selected": selected,
        "selected_arm": selected["arm"] if selected else None,
        "candidates": candidates,
        "skipped_sources": skipped,
        "common_metric_protocol": (
            "T1 macro-F1; T2 mean math equivalence; TRAPI semantic T3; "
            "T5 joint equivalence; T6 macro-F1 and positive recall; "
            "TRAPI MalRule and identical-query Eedi diagnosis-C4"
        ),
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "raw_history_visible_to_decoder": False,
        "state_token_count": 8,
        "answer_C4_confirmation_pending": selected is not None,
        "full_exact_epoch_pending": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", nargs=2, action="append", required=True,
        metavar=("SOURCE", "PATH"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(
        [(source, read(Path(path))) for source, path in args.selection]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
