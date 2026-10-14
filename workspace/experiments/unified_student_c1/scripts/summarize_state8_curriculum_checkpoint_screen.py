#!/usr/bin/env python3
"""Summarize unified-task and semantic C4 checkpoint diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from task_taxonomy_v2 import public_task_metrics, taxonomy_receipt
except ImportError:  # package-style tests
    from scripts.task_taxonomy_v2 import public_task_metrics, taxonomy_receipt


TASK_METRICS = {
    "predict_correctness": "macro_f1",
    "predict_response": "math_equivalence",
    "predict_prospective_error": "joint_math_equivalence",
    "predict_error_persistence": "macro_f1",
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def weighted(groups: dict[str, Any], prefix: str, metric: str) -> dict[str, Any]:
    rows = [value for name, value in groups.items() if name.startswith(prefix + "/")]
    if not rows or any(metric not in row for row in rows):
        raise ValueError(f"Missing {prefix}/{metric}")
    support = sum(int(row["n"]) for row in rows)
    return {
        "metric": metric,
        "score": sum(float(row[metric]) * int(row["n"]) for row in rows) / support,
        "support": support,
    }


def optional_weighted(
    groups: dict[str, Any], prefix: str, metric: str
) -> dict[str, Any] | None:
    rows = [value for name, value in groups.items() if name.startswith(prefix + "/")]
    if not rows or any(metric not in row for row in rows):
        return None
    return weighted(groups, prefix, metric)


def summarize(
    arm: str,
    step: int,
    formal_path: Path,
    t3_path: Path,
    answer_path: Path,
    malrule_path: Path,
    eedi_path: Path,
) -> dict[str, Any]:
    formal = read(formal_path)
    t3 = read(t3_path)
    answer = read(answer_path)
    malrule = read(malrule_path)
    eedi = read(eedi_path)
    groups = formal.get("metrics_by_group") or {}
    judged = t3.get("metrics_by_group") or {}
    if t3.get("judge_backend") != "trapi" or t3.get("judge_errors") != 0:
        raise ValueError("T3 semantic judging is not authoritative and complete")
    t3_rows = list(judged.values())
    t3_support = sum(int(row["llm_judge_n"]) for row in t3_rows)
    if t3_support != 75:
        raise ValueError(f"Checkpoint T3 support drift: {t3_support}")
    semantic_accuracy = sum(
        float(row["llm_judge_accuracy"]) * int(row["llm_judge_n"])
        for row in t3_rows
    ) / t3_support
    eedi_rows = [
        row for name, row in judged.items()
        if "eedi_mining/open_taxonomy_generation" in name
    ]
    if len(eedi_rows) != 1:
        raise ValueError("Eedi T3 group drift")
    task_metrics = {
        task: weighted(groups, task, metric)
        for task, metric in TASK_METRICS.items()
    }
    t6_positive_recall = optional_weighted(
        groups, "predict_error_persistence", "positive_recall"
    )
    if t6_positive_recall is None:
        raise ValueError("Missing predict_error_persistence/positive_recall")
    answer_summary = answer["summary"]
    clean = answer_summary["clean"]
    result = {
        "stage": "state8_curriculum_checkpoint_unified_semantic_summary_v1",
        "status": "complete",
        "diagnostic_only": True,
        "arm": arm,
        "checkpoint_step": step,
        "state_shape": formal["state_shape"],
        "artifact_sha256": formal["artifact_sha256"],
        "raw_history_visible_to_decoder": False,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "task_metrics": task_metrics,
        "public_task_metrics": public_task_metrics(task_metrics),
        "task_taxonomy": taxonomy_receipt(),
        "T5_positive_recall": t6_positive_recall,
        # Deprecated compatibility alias for v1 checkpoint selectors that may
        # already be running against immutable historical receipts.
        "T6_positive_recall": t6_positive_recall,
        "T3_semantic": {
            "accuracy": semantic_accuracy,
            "eedi_accuracy": float(eedi_rows[0]["llm_judge_accuracy"]),
            "support": t3_support,
            "metrics_by_group": judged,
        },
        "answer_C4": {
            "clean_n": answer_summary["clean_n"],
            "swap": clean["swap"]["flip_to_A_rate"],
            "random": clean["random"]["flip_to_A_rate"],
            "same_rule": clean["samemal"]["flip_to_A_rate"],
            "noop": clean["noop"]["flip_to_A_rate"],
            "D_specific": answer_summary["D_specific"],
            "pass": answer_summary["scientific_gate_pass"],
        },
        "malrule_diagnosis_C4": malrule,
        "eedi_same_question_C4": eedi,
    }
    if malrule.get("generation", {}).get("artifact_sha256") != result["artifact_sha256"]:
        raise ValueError("MalRule diagnosis-C4 artifact drift")
    if eedi.get("generation", {}).get("artifact_sha256") != result["artifact_sha256"]:
        raise ValueError("Eedi diagnosis-C4 artifact drift")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--t3", type=Path, required=True)
    parser.add_argument("--answer", type=Path, required=True)
    parser.add_argument("--malrule", type=Path, required=True)
    parser.add_argument("--eedi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.arm, args.step, args.formal, args.t3, args.answer, args.malrule,
        args.eedi,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
