#!/usr/bin/env python3
"""Joint promotion gate for strong1 State8 rescue arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from select_state8_strong1_widen8_bridge import (
        T3_GROUPS,
        REQUIRED_T3_RUBRIC,
        task_mean,
        task_weighted,
    )
except ImportError:
    from scripts.select_state8_strong1_widen8_bridge import (
        T3_GROUPS,
        REQUIRED_T3_RUBRIC,
        task_mean,
        task_weighted,
    )


ARMS = {
    "taxonomy_joint": {
        "job": "state8-strong1-rescue-taxonomy_joint-seed61",
        "finalfit_variant": "strong1_rescue_taxonomy_joint_seed61",
        "finalfit_job": "state8-strong1-rescue-taxonomy-joint-seed61-unified-finalfit-seed42",
        "stable_root": "/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed61_trajectory01_strong1_rescue_taxonomy_joint_v1",
    },
    "taskanchor_c4strong": {
        "job": "state8-strong1-rescue-taskanchor_c4strong-seed62",
        "finalfit_variant": "strong1_rescue_taskanchor_c4strong_seed62",
        "finalfit_job": "state8-strong1-rescue-taskanchor-c4strong-seed62-unified-finalfit-seed42",
        "stable_root": "/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed62_taskanchor_strong1_rescue_taskanchor_c4strong_v1",
    },
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def one(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} below {root}, observed={paths}")
    return paths[0]


def score(slug: str, root: Path) -> dict[str, Any]:
    if slug not in ARMS:
        raise ValueError(f"Unknown rescue arm: {slug}")
    job = ARMS[slug]["job"]
    artifact = one(root / "download", f"**/{job}").resolve()
    formal = read(artifact / "eval/formal_t3_300/metrics.json")
    publish = read(artifact / "metrics/publish_receipt.json")
    answer = read(artifact / "eval/answer_c4_150/recurrent_causal_gate.json")
    source = read(artifact / "metrics/strong1_source_normalized_receipt.json")
    t3 = read(root / slug / "t3_semantic/metrics.json")
    diagnosis = read(root / slug / "c4_semantic/summary.json")

    artifacts = publish.get("artifact_sha256") or {}
    for name, value in (
        ("formal", formal.get("artifact_sha256")),
        ("answer", answer.get("artifact_sha256")),
        ("diagnosis", (diagnosis.get("generation") or {}).get("artifact_sha256")),
    ):
        if value != artifacts:
            raise ValueError(f"Rescue artifact binding drift: {slug}/{name}")
    if publish.get("stage") != "state8_stateonly_repair_pilot_publish_v1":
        raise ValueError(f"Rescue publish stage drift: {slug}")
    expected_entrypoint = (
        "scripts/train_state8_diagnosis_first_causal_repair.py"
        if slug == "taxonomy_joint"
        else "scripts/train_state8_task_anchored_causal_repair.py"
    )
    if publish.get("train_entrypoint") != expected_entrypoint:
        raise ValueError(f"Rescue entrypoint drift: {slug}")
    if source.get("normalized_from_stage") is None or source.get("strong1_source_slug") is None:
        raise ValueError(f"Rescue source provenance drift: {slug}")
    for name, value in (("formal", formal), ("answer", answer)):
        if value.get("state_shape") != [64, 8, 5120]:
            raise ValueError(f"Rescue State8 drift: {slug}/{name}")
        if value.get("raw_history_visible_to_decoder") is not False:
            raise ValueError(f"Rescue history leakage: {slug}/{name}")
        if value.get("persistent_history_kv_cache") is not False:
            raise ValueError(f"Rescue history cache drift: {slug}/{name}")
    if t3.get("judge_backend") != "trapi" or int(t3.get("judge_errors", -1)) != 0:
        raise ValueError(f"Incomplete rescue T3 judgment: {slug}")
    if t3.get("judge_rubric") != REQUIRED_T3_RUBRIC:
        raise ValueError(f"Stale or non-substantive rescue T3 rubric: {slug}")
    groups = t3.get("metrics_by_group") or {}
    if set(groups) != set(T3_GROUPS):
        raise ValueError(f"Rescue T3 group drift: {slug}")
    values: dict[str, float] = {}
    schema_numerator = 0.0
    support = 0
    for name, short in T3_GROUPS.items():
        row = groups[name]
        n = int(row.get("llm_judge_n", -1))
        if n <= 0 or int(row.get("llm_judge_exact_shortcuts", -1)) != 0:
            raise ValueError(f"Rescue T3 authority drift: {slug}/{short}")
        if int(row.get("llm_judge_schema_valid_n", -1)) < 0:
            raise ValueError(f"Rescue substantive-schema audit missing: {slug}/{short}")
        if "llm_judge_raw_semantic_accuracy" not in row:
            raise ValueError(f"Rescue raw semantic audit missing: {slug}/{short}")
        values[short] = float(row["llm_judge_accuracy"])
        schema_numerator += float(row["natural_schema_valid_rate"]) * n
        support += n
    synthetic = 0.5 * (values["cross"] + values["same"])
    schema = schema_numerator / support
    formal_groups = formal.get("metrics_by_group") or {}
    answer_summary = answer.get("summary") or {}
    answer_clean = answer_summary.get("clean") or {}
    diagnosis_cells = diagnosis.get("cells") or {}
    checks = {
        "T1": task_mean(formal_groups, "predict_correctness/", "accuracy"),
        "T2": task_mean(formal_groups, "predict_response/", "math_equivalence"),
        "T3_eedi_semantic": values["eedi"],
        "T3_synthetic_semantic": synthetic,
        "T3_schema": schema,
        "T5": task_mean(formal_groups, "predict_prospective_error/", "joint_math_equivalence"),
        "T6": task_mean(formal_groups, "predict_error_persistence/", "accuracy"),
        "T6_macro_f1": task_weighted(formal_groups, "predict_error_persistence/", "macro_f1"),
        "T6_positive_recall": task_weighted(formal_groups, "predict_error_persistence/", "positive_recall"),
        "answer_C4_pass": bool(answer_summary.get("scientific_gate_pass")),
        "answer_C4_D_specific": float(answer_summary.get("D_specific", -1.0)),
        "answer_C4_swap": float((answer_clean.get("swap") or {}).get("flip_to_A_rate", -1.0)),
        "answer_C4_random": float((answer_clean.get("random") or {}).get("flip_to_A_rate", 1.0)),
        "answer_C4_same_rule": float((answer_clean.get("samemal") or {}).get("flip_to_A_rate", 1.0)),
        "answer_C4_noop": float((answer_clean.get("noop") or {}).get("flip_to_A_rate", 1.0)),
        "diagnosis_C4_pass": bool(diagnosis.get("pass")),
        "diagnosis_C4_D_specific": float(diagnosis.get("semantic_D_specific", -1.0)),
        "diagnosis_C4_swap_to_donor": float((diagnosis_cells.get("swap_to_donor") or {}).get("semantic_equivalence_rate", -1.0)),
        "diagnosis_C4_same_rule_retention": float(diagnosis.get("same_rule_recipient_retention", -1.0)),
        "diagnosis_C4_noop_retention": float(diagnosis.get("noop_recipient_retention", -1.0)),
        "diagnosis_C4_random_stable": float(diagnosis.get("random_matches_either_stable_target_rate", 1.0)),
    }
    floors = {
        "T1": 0.55, "T2": 0.20, "T3_eedi_semantic": 0.17,
        "T3_synthetic_semantic": 0.45, "T3_schema": 0.95,
        "T5": 0.35, "T6": 0.35, "T6_macro_f1": 0.30,
        "T6_positive_recall": "strictly_greater_than_zero",
        "answer_C4_D_specific": 0.15, "answer_C4_same_rule_max": 0.10,
        "answer_C4_noop_max": 0.05, "diagnosis_C4_D_specific": 0.15,
        "diagnosis_C4_swap_to_donor": 0.50,
        "diagnosis_C4_same_rule_retention": 0.65,
        "diagnosis_C4_noop_retention": 0.75,
        "diagnosis_C4_random_stable_max": 0.25,
    }
    qualified = (
        checks["answer_C4_pass"] and checks["diagnosis_C4_pass"]
        and all(checks[key] >= floors[key] for key in (
            "T1", "T2", "T3_eedi_semantic", "T3_synthetic_semantic", "T3_schema",
            "T5", "T6", "T6_macro_f1", "answer_C4_D_specific",
            "diagnosis_C4_D_specific", "diagnosis_C4_swap_to_donor",
            "diagnosis_C4_same_rule_retention", "diagnosis_C4_noop_retention",
        ))
        and checks["T6_positive_recall"] > 0.0
        and checks["answer_C4_same_rule"] <= floors["answer_C4_same_rule_max"]
        and checks["answer_C4_noop"] <= floors["answer_C4_noop_max"]
        and checks["diagnosis_C4_random_stable"] <= floors["diagnosis_C4_random_stable_max"]
    )
    aggregate = (
        .10 * checks["T1"] + .08 * checks["T2"]
        + .12 * checks["T3_eedi_semantic"] + .12 * checks["T3_synthetic_semantic"]
        + .05 * checks["T3_schema"] + .08 * checks["T5"]
        + .05 * checks["T6_macro_f1"] + .15 * checks["answer_C4_D_specific"]
        + .15 * checks["diagnosis_C4_D_specific"]
        + .10 * checks["diagnosis_C4_swap_to_donor"]
        - .10 * checks["diagnosis_C4_random_stable"]
    )
    return {
        "slug": slug, "qualified": qualified, "score": aggregate,
        "checks": checks, "floors": floors, "artifact_sha256": artifacts,
        "remote_root": ARMS[slug]["stable_root"],
        "finalfit_variant": ARMS[slug]["finalfit_variant"],
        "finalfit_job": ARMS[slug]["finalfit_job"],
        "source_provenance": source,
        "exact_match_authoritative": False, "semantic_judge_authoritative": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [score(slug, args.root) for slug in ARMS]
    eligible = [row for row in candidates if row["qualified"]]
    selected = max(eligible, key=lambda row: row["score"]) if eligible else None
    result = {
        "stage": "state8_strong1_rescue_joint_promotion_v1",
        "pass": selected is not None, "selected": selected,
        "candidates": sorted(candidates, key=lambda row: row["slug"]),
        "selection_scope": "strict_state_only_unified_plus_answer_and_natural_diagnosis_C4",
        "exact_match_authoritative": False, "semantic_judge_authoritative": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if selected else 2)


if __name__ == "__main__":
    main()
