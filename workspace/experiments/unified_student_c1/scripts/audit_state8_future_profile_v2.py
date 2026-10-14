#!/usr/bin/env python3
"""Fail-closed independent audit of the State8 future-profile v2 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TAGS = {
    name: re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.I | re.S)
    for name in ("weakness", "strength", "profile", "evidence_confidence")
}
CONDITIONS = {"native", "noop", "same_profile", "random", "swap"}
TASKS = {
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "predict_prospective_error",
    "predict_error_persistence",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def values(text: str, tag: str) -> list[str]:
    match = TAGS[tag].search(str(text))
    if not match:
        return []
    value = match.group(1).strip()
    if value.casefold() == "insufficient_evidence":
        return ["insufficient_evidence"]
    return [piece.strip() for piece in value.split(";") if piece.strip()]


def profile_key(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(values(text, "weakness")), tuple(values(text, "strength"))


def set_f1(predicted: list[str], gold: list[str]) -> float:
    p, g = set(predicted), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = len(p & g)
    return 2 * overlap / (len(p) + len(g))


def similarity(predicted: str, target: str) -> float:
    return 0.5 * (
        set_f1(values(predicted, "weakness"), values(target, "weakness"))
        + set_f1(values(predicted, "strength"), values(target, "strength"))
    )


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def audit_profiles(metrics: dict[str, Any], rows: list[dict[str, Any]], failures: list[str]) -> None:
    check(metrics.get("stage") == "state8_future_student_profile_end_to_end_readout_v2", "profile stage is not v2", failures)
    check(metrics.get("status") == "complete", "profile metrics are not complete", failures)
    check(len(rows) == 88, f"expected 88 profile rows, found {len(rows)}", failures)
    check({row.get("profile_index") for row in rows} == set(range(1, 89)), "profile indices are incomplete or duplicated", failures)
    schema = [all(pattern.search(str(row.get("output", ""))) for pattern in TAGS.values()) for row in rows]
    exact = [profile_key(row.get("output", "")) == profile_key(row.get("gold", "")) for row in rows]
    future_f1 = [
        set_f1(values(row.get("output", ""), "weakness"), values(row.get("gold", ""), "weakness"))
        for row in rows
    ]
    for index, (row, valid, exact_value, f1) in enumerate(zip(rows, schema, exact, future_f1, strict=True), start=1):
        check(row.get("schema_valid") is valid, f"profile row {index} schema flag mismatch", failures)
        check(row.get("exact_profile") is exact_value, f"profile row {index} exact flag mismatch", failures)
        check(close(row.get("future_weakness_f1", -1), f1), f"profile row {index} future F1 mismatch", failures)
        check(row.get("raw_history_visible_to_decoder") is False, f"profile row {index} exposes history", failures)
        check(row.get("student_id_visible_to_decoder") is False, f"profile row {index} exposes student id", failures)
    recomputed = {
        "n": len(rows),
        "schema_valid_rate": sum(schema) / len(rows),
        "exact_profile_accuracy": sum(exact) / len(rows),
        "future_weakness_macro_f1": sum(future_f1) / len(rows),
        "observed_weakness_macro_f1": sum(float(row["observed_weakness_f1"]) for row in rows) / len(rows),
        "observed_strength_macro_f1": sum(float(row["observed_strength_f1"]) for row in rows) / len(rows),
    }
    for name, value in recomputed.items():
        check(close(metrics["profile"].get(name, -1), value), f"profile aggregate {name} mismatch", failures)
    expected_gate = recomputed["schema_valid_rate"] >= 0.95 and recomputed["future_weakness_macro_f1"] >= 0.60
    check(metrics.get("scientific_gate_pass") is expected_gate, "profile scientific gate mismatch", failures)
    check(metrics.get("raw_history_visible_to_decoder") is False, "profile metrics expose history", failures)
    check(metrics.get("student_id_visible_to_decoder") is False, "profile metrics expose student id", failures)


def audit_transport(metrics: dict[str, Any], rows: list[dict[str, Any]], failures: list[str]) -> None:
    counts = Counter(str(row.get("condition")) for row in rows)
    check(len(rows) == 250, f"expected 250 transport rows, found {len(rows)}", failures)
    check(counts == Counter({condition: 50 for condition in CONDITIONS}), f"transport condition counts differ: {dict(counts)}", failures)
    pair_conditions: dict[str, set[str]] = defaultdict(set)
    recomputed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        condition = str(row.get("condition"))
        pair_conditions[str(row.get("pair_id"))].add(condition)
        native = similarity(row.get("output", ""), row.get("native_target", ""))
        donor = similarity(row.get("output", ""), row.get("donor_target", ""))
        key = profile_key(row.get("output", ""))
        category = (
            "neutral" if key == (("insufficient_evidence",), ("insufficient_evidence",))
            else "donor" if donor > native and donor >= 0.5
            else "recipient" if native > donor and native >= 0.5
            else "other"
        )
        check(close(row.get("native_profile_similarity", -1), native), f"transport row {index} native similarity mismatch", failures)
        check(close(row.get("donor_profile_similarity", -1), donor), f"transport row {index} donor similarity mismatch", failures)
        check(row.get("category") == category, f"transport row {index} category mismatch", failures)
        check(row.get("query_held_fixed") is True, f"transport row {index} query was not held fixed", failures)
        check(row.get("only_state_identity_changes") is True, f"transport row {index} changes more than state", failures)
        recomputed[condition].append({"category": category})
    check(len(pair_conditions) == 50, f"expected 50 transport pairs, found {len(pair_conditions)}", failures)
    check(all(value == CONDITIONS for value in pair_conditions.values()), "some transport pairs lack a condition", failures)
    summary = {}
    for condition in CONDITIONS:
        items = recomputed[condition]
        summary[condition] = {
            "n": len(items),
            "to_donor_rate": sum(item["category"] == "donor" for item in items) / len(items),
            "to_recipient_rate": sum(item["category"] == "recipient" for item in items) / len(items),
            "neutral_rate": sum(item["category"] == "neutral" for item in items) / len(items),
        }
        for name, value in summary[condition].items():
            check(close(metrics["transport"]["by_condition"][condition].get(name, -1), value), f"transport {condition}/{name} mismatch", failures)
    d_specific = summary["swap"]["to_donor_rate"] - summary["random"]["to_donor_rate"]
    d_same = summary["native"]["to_recipient_rate"] - summary["same_profile"]["to_recipient_rate"]
    check(close(metrics["transport"].get("D_profile_specific", 99), d_specific), "D_profile_specific mismatch", failures)
    check(close(metrics["transport"].get("D_profile_same_control", 99), d_same), "D_profile_same_control mismatch", failures)


def interface_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    uncertainty = metrics["uncertainty"]
    comparison = uncertainty["state8_vs_population"]
    history_comparison = uncertainty["state8_vs_same_window_kc_outcomes"]
    return {
        "heldout_student_support": metrics["students"]["test"] >= 80,
        "predictive_utility_rmse": comparison["rmse_delta_model_minus_reference"]["bootstrap_probability_model_better"] > 0.95,
        "readable_weakness_utility": comparison["weak_f1_delta_model_minus_reference"]["bootstrap_probability_model_better"] > 0.95,
        "predictive_utility_beyond_same_window_rmse": history_comparison[
            "rmse_delta_model_minus_reference"
        ]["bootstrap_probability_model_better"] > 0.95,
        "readable_weakness_beyond_same_window": history_comparison[
            "weak_f1_delta_model_minus_reference"
        ]["bootstrap_probability_model_better"] > 0.95,
        "state_ownership_matters": uncertainty["state_ownership_permutation_null"]["rmse"]["p_one_sided"] < 0.05,
        "same_student_disjoint_history_stability": uncertainty["disjoint_history_prediction_stability"]["p_one_sided"] < 0.05,
        "closed_information_route": metrics.get("raw_history_visible_to_readout") is False and metrics.get("student_id_visible_to_readout") is False,
    }


def audit_interface(profile: dict[str, Any], metrics: dict[str, Any], root: Path, failures: list[str]) -> None:
    check(metrics.get("status") == "complete", "interface metrics are not complete", failures)
    check(metrics.get("students") == {"train": 663, "valid": 84, "test": 88}, "interface student split differs", failures)
    check(metrics.get("student_disjoint") is True, "interface split is not student-disjoint", failures)
    check(metrics.get("future_support_visible_to_readout") is False, "interface exposes future support", failures)
    check(metrics.get("weakness_candidate_protocol") == "fixed_train_valid_supported_kcs", "interface weakness candidates are not fitting-split fixed", failures)
    gates = interface_gates(metrics)
    check(metrics.get("gates") == gates, "interface gate values do not recompute", failures)
    check(metrics.get("scientific_gate_pass") is all(gates.values()), "interface scientific gate mismatch", failures)
    expected_checkpoint = (profile["artifact_sha256"]["adapter_model"], profile["artifact_sha256"]["recurrent_state"])
    for split in ("train", "valid", "test"):
        entry = metrics["feature_artifacts"][split]
        checkpoint = (entry["checkpoint"]["adapter"], entry["checkpoint"]["recurrent_state"])
        check(checkpoint == expected_checkpoint, f"interface {split} checkpoint hash differs from profile", failures)
        npz = root / f"{split}.npz"
        metadata = root / f"{split}.json"
        check(npz.is_file(), f"missing interface feature {npz}", failures)
        check(metadata.is_file(), f"missing interface metadata {metadata}", failures)
        if npz.is_file():
            check(sha256(npz) == entry["npz_sha256"], f"interface {split} npz hash mismatch", failures)
        if metadata.is_file():
            check(sha256(metadata) == entry["metadata_sha256"], f"interface {split} metadata hash mismatch", failures)


def audit_retention(profile: dict[str, Any], metrics: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    check(metrics.get("raw_history_visible_to_decoder") is False, "retention decoder exposes raw history", failures)
    check(metrics.get("persistent_history_kv_cache") is False, "retention keeps history KV cache", failures)
    check(metrics.get("state_shape") == [64, 8, 5120], "retention state shape differs", failures)
    check(metrics.get("artifact_sha256") == profile.get("artifact_sha256"), "retention checkpoint hash differs from profile", failures)
    groups = metrics.get("metrics_by_group", {})
    tasks = {str(name).split("/", 1)[0] for name in groups}
    check(tasks == TASKS, f"retention task coverage differs: {sorted(tasks)}", failures)
    total = sum(int(group.get("n", 0)) for group in groups.values())
    check(total == metrics.get("sampled_rows") == 200, f"retention sample cardinality differs: {total}", failures)
    pending = [name for name, group in groups.items() if group.get("full_taxonomy_retrieval_pending")]
    return {"groups": len(groups), "rows": total, "tasks": sorted(tasks), "pending_full_evaluation": pending}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-metrics", type=Path, required=True)
    parser.add_argument("--profiles-jsonl", type=Path, required=True)
    parser.add_argument("--transport-jsonl", type=Path, required=True)
    parser.add_argument("--interface-metrics", type=Path, required=True)
    parser.add_argument("--interface-feature-dir", type=Path, required=True)
    parser.add_argument("--retention-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile_metrics = read_json(args.profile_metrics)
    profiles = read_jsonl(args.profiles_jsonl)
    transport = read_jsonl(args.transport_jsonl)
    interface = read_json(args.interface_metrics)
    retention = read_json(args.retention_metrics)
    failures: list[str] = []
    audit_profiles(profile_metrics, profiles, failures)
    audit_transport(profile_metrics, transport, failures)
    audit_interface(profile_metrics, interface, args.interface_feature_dir, failures)
    retention_summary = audit_retention(profile_metrics, retention, failures)
    result = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "profile_scientific_gate_pass": profile_metrics.get("scientific_gate_pass"),
        "interface_scientific_gate_pass": interface.get("scientific_gate_pass"),
        "interface_gates": interface.get("gates"),
        "retention": retention_summary,
        "artifact_sha256": profile_metrics.get("artifact_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
