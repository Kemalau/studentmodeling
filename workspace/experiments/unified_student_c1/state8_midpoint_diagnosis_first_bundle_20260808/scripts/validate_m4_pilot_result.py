#!/usr/bin/env python3
"""Fail closed on the strict-state M4 pilot before the 259845-row final fit.

The AMLT job state is only an engineering signal.  This gate additionally
requires all five formal task families, a closed decoder route, non-collapsed
task behaviour, and a causal-specific C4 state intervention on the frozen
150-pair panel.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_TASKS = {
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "predict_prospective_error",
    "predict_error_persistence",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def finite(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {location}: {value}")


def weighted(groups: list[dict[str, Any]], field: str) -> float:
    supported = [group for group in groups if group.get(field) is not None]
    denominator = sum(int(group.get("n", 0)) for group in supported)
    if denominator <= 0:
        raise ValueError(f"No support for metric {field}")
    return sum(float(group[field]) * int(group["n"]) for group in supported) / denominator


def support(groups: list[dict[str, Any]]) -> int:
    return sum(int(group.get("n", 0)) for group in groups)


def find_formal_metrics(root: Path) -> tuple[list[Path], dict[str, list[dict[str, Any]]]]:
    paths: list[Path] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_groups: set[str] = set()
    for path in sorted(root.rglob("metrics.json")):
        payload = read_json(path)
        metrics = payload.get("metrics_by_group")
        if not isinstance(metrics, dict) or not metrics:
            continue
        tasks = {str(name).split("/", 1)[0] for name in metrics}
        if not tasks.intersection(EXPECTED_TASKS):
            continue
        if payload.get("information_route") != (
            "sequential_updates_then_continuous_state_plus_current_query"
        ):
            raise ValueError(f"Non-recurrent information route in {path}")
        if payload.get("raw_history_visible_to_decoder") is not False:
            raise ValueError(f"Raw-history leak in {path}")
        if payload.get("persistent_history_kv_cache") is not False:
            raise ValueError(f"Persistent history KV cache in {path}")
        paths.append(path)
        for name, values in metrics.items():
            task = str(name).split("/", 1)[0]
            if task not in EXPECTED_TASKS:
                continue
            if name in seen_groups:
                raise ValueError(f"Duplicate formal metric group {name}")
            seen_groups.add(name)
            if not isinstance(values, dict):
                raise ValueError(f"Invalid metric group {name} in {path}")
            finite(values, name)
            by_task[task].append(values)
    if set(by_task) != EXPECTED_TASKS:
        raise ValueError(
            f"Formal task coverage mismatch: expected={sorted(EXPECTED_TASKS)} "
            f"got={sorted(by_task)}"
        )
    return paths, by_task


def find_c4(root: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted(root.rglob("recurrent_causal_gate.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one recurrent C4 result, found {paths}")
    payload = read_json(paths[0])
    if payload.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("C4 decoder saw raw history")
    if payload.get("persistent_history_kv_cache") is not False:
        raise ValueError("C4 decoder used persistent history KV cache")
    if payload.get("intervention_target") != "final_continuous_recurrent_state":
        raise ValueError("C4 intervention did not target the persistent recurrent state")
    return paths[0], payload


def validate(root: Path) -> dict[str, Any]:
    formal_paths, by_task = find_formal_metrics(root)
    c4_path, c4 = find_c4(root)
    summary = c4.get("summary") or {}
    clean = summary.get("clean") or {}

    aggregates = {
        "T1_accuracy": weighted(by_task["predict_correctness"], "accuracy"),
        "T1_macro_f1": weighted(by_task["predict_correctness"], "macro_f1"),
        "T2_math_equivalence": weighted(by_task["predict_response"], "math_equivalence"),
        "T2_format_valid": weighted(by_task["predict_response"], "format_valid_rate"),
        "T3_format_valid": weighted(by_task["diagnose_misconception"], "format_valid_rate"),
        "T5_joint_math_equivalence": weighted(
            by_task["predict_prospective_error"], "joint_math_equivalence"
        ),
        "T5_format_valid": weighted(
            by_task["predict_prospective_error"], "format_valid_rate"
        ),
        "T6_macro_f1": weighted(by_task["predict_error_persistence"], "macro_f1"),
        "T6_auroc": weighted(by_task["predict_error_persistence"], "auroc"),
        "C4_D_specific": float(summary.get("D_specific", float("nan"))),
        "C4_D_damage": float(summary.get("D_damage", float("nan"))),
    }
    finite(aggregates, "aggregates")
    supports = {task: support(groups) for task, groups in by_task.items()}
    checks = {
        "four_formal_partitions": len(formal_paths) == 4,
        "T1_support": supports["predict_correctness"] >= 500,
        "T2_support": supports["predict_response"] >= 300,
        "T3_support": supports["diagnose_misconception"] >= 300,
        "T5_support": supports["predict_prospective_error"] >= 300,
        "T6_support": supports["predict_error_persistence"] >= 100,
        "T1_noncollapsed": aggregates["T1_accuracy"] >= 0.50
        and aggregates["T1_macro_f1"] >= 0.40,
        "T2_noncollapsed": aggregates["T2_math_equivalence"] >= 0.25
        and aggregates["T2_format_valid"] >= 0.80,
        # Exact string match is not a valid primary metric for open-taxonomy T3.
        # The pilot only gates formatting; the final report adds an independent
        # semantic judge before making a T3 quality claim.
        "T3_noncollapsed": aggregates["T3_format_valid"] >= 0.80,
        "T5_noncollapsed": aggregates["T5_joint_math_equivalence"] >= 0.35
        and aggregates["T5_format_valid"] >= 0.80,
        "T6_noncollapsed": aggregates["T6_macro_f1"] >= 0.35
        and aggregates["T6_auroc"] >= 0.45,
        "C4_support": int(summary.get("clean_n", 0)) >= 50,
        "C4_specificity": aggregates["C4_D_specific"] >= 0.15,
        "C4_noop": float((clean.get("noop") or {}).get("flip_to_A_rate", 1.0)) <= 0.05,
        "C4_same_rule": float(
            (clean.get("samemal") or {}).get("flip_to_A_rate", 1.0)
        )
        <= 0.10,
        "C4_declared_gate": summary.get("scientific_gate_pass") is True,
    }
    payload = {
        "stage": "M4_strict_recurrent_pilot",
        "pass": all(checks.values()),
        "checks": checks,
        "supports": supports,
        "aggregates": aggregates,
        "formal_metrics": [str(path) for path in formal_paths],
        "c4_result": str(c4_path),
        "c4_summary": summary,
        "t3_primary_metric_pending": "independent semantic judge after final fit",
    }
    if not payload["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"M4 pilot failed scientific checks: {failed}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
