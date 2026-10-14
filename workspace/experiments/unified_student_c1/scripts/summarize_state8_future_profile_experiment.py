#!/usr/bin/env python3
"""Build a compact, fail-closed State8 future-profile experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_METRICS = {
    "predict_correctness": ("accuracy", "auroc", "auprc"),
    "predict_response": ("math_equivalence", "normalized_exact_match", "format_valid_rate"),
    "diagnose_misconception": ("normalized_exact_match", "natural_schema_valid_rate", "format_valid_rate"),
    "predict_prospective_error": ("joint_math_equivalence", "correctness_accuracy", "format_valid_rate"),
    "predict_error_persistence": ("accuracy", "auroc", "auprc"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_task_summary(retention: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for name, metrics in retention.get("metrics_by_group", {}).items():
        task = str(name).split("/", 1)[0]
        grouped.setdefault(task, []).append(metrics)
    output: dict[str, Any] = {}
    for task, rows in sorted(grouped.items()):
        total = sum(int(row.get("n", 0)) for row in rows)
        entry: dict[str, Any] = {"groups": len(rows), "n": total}
        for metric in TASK_METRICS.get(task, ()):
            available = [row for row in rows if row.get(metric) is not None]
            denominator = sum(int(row.get("n", 0)) for row in available)
            if denominator:
                entry[metric] = sum(
                    float(row[metric]) * int(row.get("n", 0)) for row in available
                ) / denominator
                entry[f"{metric}_n"] = denominator
        entry["full_evaluation_pending_groups"] = sum(
            bool(row.get("full_taxonomy_retrieval_pending")) for row in rows
        )
        output[task] = entry
    return output


def build(
    profile: dict[str, Any],
    interface: dict[str, Any],
    retention: dict[str, Any],
    audit: dict[str, Any],
    experiment: str,
) -> dict[str, Any]:
    state = interface["metrics"]["state8"]
    population = interface["metrics"]["train_valid_population"]
    same_window = interface["metrics"]["same_window_kc_outcomes"]
    permuted = interface["metrics"]["permuted_state_ownership"]
    uncertainty = interface["uncertainty"]
    ownership = uncertainty["state_ownership_permutation_null"]
    result = {
        "experiment": experiment,
        "audit": {
            "status": audit.get("status"),
            "failures": audit.get("failures", []),
            "artifact_sha256": audit.get("artifact_sha256"),
        },
        "protocol": {
            "strict_state_only": (
                profile.get("raw_history_visible_to_decoder") is False
                and profile.get("student_id_visible_to_decoder") is False
                and interface.get("raw_history_visible_to_readout") is False
                and interface.get("student_id_visible_to_readout") is False
            ),
            "student_disjoint": interface.get("student_disjoint"),
            "heldout_students": interface.get("students", {}).get("test"),
            "state_token_count": interface.get("state_token_count"),
            "weakness_candidate_protocol": interface.get("weakness_candidate_protocol"),
        },
        "readability": profile.get("profile", {}),
        "reusable_readout": {
            "state8_future_rate_rmse": state["future_rate"]["rmse"],
            "state8_weakness_macro_f1": state["weakness"]["weak_top2_macro_f1"],
            "population_future_rate_rmse": population["future_rate"]["rmse"],
            "population_weakness_macro_f1": population["weakness"]["weak_top2_macro_f1"],
            "same_window_future_rate_rmse": same_window["future_rate"]["rmse"],
            "same_window_weakness_macro_f1": same_window["weakness"]["weak_top2_macro_f1"],
        },
        "ownership": {
            "observed_rmse": ownership["rmse"]["observed"],
            "permutation_p_one_sided": ownership["rmse"]["p_one_sided"],
            "permutation_null_ci95": ownership["rmse"]["null_ci95"],
            "deterministic_permuted_rmse": permuted["future_rate"]["rmse"],
            "deterministic_permuted_minus_native_rmse": (
                permuted["future_rate"]["rmse"] - state["future_rate"]["rmse"]
            ),
        },
        "stability": uncertainty["disjoint_history_prediction_stability"],
        "transport_diagnostic": profile.get("transport", {}),
        "c4_gates": interface.get("gates", {}),
        "c4_scientific_gate_pass": interface.get("scientific_gate_pass") is True,
        "profile_scientific_gate_pass": profile.get("scientific_gate_pass") is True,
        "task_performance": weighted_task_summary(retention),
        "task_performance_scope": {
            "sampled_rows": retention.get("sampled_rows"),
            "groups": len(retention.get("metrics_by_group", {})),
            "is_retention_delta": False,
            "note": "Post-training same-protocol performance; no matched pre-training score was supplied.",
        },
    }
    result["experiment_complete"] = (
        result["audit"]["status"] == "pass"
        and result["protocol"]["strict_state_only"]
        and result["protocol"]["student_disjoint"] is True
        and result["task_performance_scope"]["sampled_rows"] == 200
    )
    result["scientific_conclusion"] = (
        "c4_supported" if result["c4_scientific_gate_pass"]
        else "experiment_complete_but_c4_not_supported"
    )
    return result


def markdown(result: dict[str, Any]) -> str:
    readability = result["readability"]
    readout = result["reusable_readout"]
    ownership = result["ownership"]
    lines = [
        f"# {result['experiment']}", "",
        f"- Experiment complete: `{str(result['experiment_complete']).lower()}`",
        f"- Independent audit: `{result['audit']['status']}`",
        f"- Scientific conclusion: `{result['scientific_conclusion']}`",
        f"- Strict state-only: `{str(result['protocol']['strict_state_only']).lower()}`",
        "",
        "## New C4 and readability", "",
        f"- Natural profile schema-valid rate: {readability.get('schema_valid_rate', 0):.4f}",
        f"- Natural future-weakness macro-F1: {readability.get('future_weakness_macro_f1', 0):.4f}",
        f"- Reusable-readout future RMSE: {readout['state8_future_rate_rmse']:.6f}",
        f"- Reusable-readout weakness macro-F1: {readout['state8_weakness_macro_f1']:.4f}",
        f"- Ownership permutation p (one-sided): {ownership['permutation_p_one_sided']:.6f}",
        f"- Deterministic permuted-minus-native RMSE: {ownership['deterministic_permuted_minus_native_rmse']:+.6f}",
        "",
        "## Task performance", "",
        "| Task | n | Primary metrics |", "|---|---:|---|",
    ]
    for task, values in result["task_performance"].items():
        metrics = ", ".join(
            f"{name}={value:.4f}" for name, value in values.items()
            if name not in {"groups", "n", "full_evaluation_pending_groups"}
            and not name.endswith("_n") and isinstance(value, (int, float))
        )
        lines.append(f"| {task} | {values['n']} | {metrics} |")
    lines += ["", result["task_performance_scope"]["note"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-metrics", type=Path, required=True)
    parser.add_argument("--interface-metrics", type=Path, required=True)
    parser.add_argument("--retention-metrics", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        read_json(args.profile_metrics), read_json(args.interface_metrics),
        read_json(args.retention_metrics), read_json(args.independent_audit),
        args.experiment,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
