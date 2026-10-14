#!/usr/bin/env python3
"""Validate a completed causal-rescue M4 artifact before formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique(root: Path, suffix: str) -> Path:
    paths = sorted(root.rglob(suffix))
    if len(paths) != 1:
        raise ValueError(f"Expected one {suffix}, found {paths}")
    return paths[0]


def validate(root: Path, branch: str) -> dict:
    completion_path = unique(root, "metrics/completion.json")
    metrics = completion_path.parent
    job = metrics.parent
    completion = load(completion_path)
    config = load(unique(root, "metrics/run_config.json"))
    c4_path = unique(root, "recurrent_causal_gate.json")
    c4 = load(c4_path)
    summary = c4.get("summary") or {}
    clean = summary.get("clean") or {}
    settings = {
        "raw4": (4.0, False, 0.0, 1726, "mean"),
        "mean_normalized": (1.0, True, 0.0, 1726, "mean"),
        "raw8": (8.0, False, 0.0, 1726, "mean"),
        "mean2_normalized": (2.0, True, 0.0, 1726, "mean"),
        "repeat100_w1": (1.0, False, 1.0, 12000, "mean"),
        "repeat100_w4": (4.0, False, 1.0, 12000, "mean"),
        "repeat100_sum_w1": (1.0, False, 1.0, 12000, "sum"),
        "repeat100_sum_w4": (4.0, False, 1.0, 12000, "sum"),
        "bootstrap_sum_w1": (1.0, False, 1.0, 12000, "sum"),
        "bootstrap_sum_w4": (4.0, False, 1.0, 12000, "sum"),
        "bootstrap_native25_sum_w1": (1.0, False, 1.0, 12000, "sum"),
        "bootstrap_native25_sum_w4": (4.0, False, 1.0, 12000, "sum", "symmetric"),
        "bootstrap_balanced_sum_w1": (1.0, False, 1.0, 12000, "sum", "directional"),
        "bootstrap_balanced_sum_w4": (4.0, False, 1.0, 12000, "sum", "directional"),
        "bootstrap_native25_balanced_sum_w1": (1.0, False, 1.0, 12000, "sum", "directional"),
        "bootstrap_native25_balanced_sum_w4": (4.0, False, 1.0, 12000, "sum", "directional"),
        "donor_directional_sum_w1": (1.0, False, 1.0, 12000, "sum", "donor_directional"),
        "donor_directional_sum_w4": (4.0, False, 1.0, 12000, "sum", "donor_directional"),
        "cf4_bootstrap_symmetric_w1": (1.0, False, 1.0, 12000, "sum", "symmetric"),
        "cf4_donor_directional_w1": (1.0, False, 1.0, 12000, "sum", "donor_directional"),
    }
    settings = {
        key: (*value, "symmetric") if len(value) == 5 else value
        for key, value in settings.items()
    }
    expected_weight, expected_normalize, expected_repeat, expected_exposures, expected_reduction, expected_mode = settings[branch]
    checks = {
        "complete": completion.get("status") == "complete" and completion.get("diagnostic") is False,
        "one_epoch": completion.get("micro_steps") == 12000 and completion.get("real_examples_seen_once") is True,
        "causal_exact": completion.get("causal_pairs_seen") == expected_exposures
        and completion.get("causal_exposure_count_matches_plan") is True,
        "causal_coverage": completion.get("all_causal_pairs_covered") is True,
        "branch_weight": float(config.get("causal_weight", -1)) == expected_weight,
        "branch_normalization": bool(config.get("normalize_causal_exposure")) == expected_normalize,
        "branch_rehearsal": float(config.get("causal_repeat_fraction", -1.0)) == expected_repeat,
        "branch_preference_reduction": config.get("causal_preference_reduction") == expected_reduction,
        "branch_objective_mode": config.get("causal_objective_mode") == expected_mode,
        "closed_decoder": config.get("decoder_view") == "state_plus_current_query_strict_late_fusion",
        "adapter_hash": completion.get("sha256", {}).get("adapter_model") == sha256(job / "model/adapter_model.safetensors"),
        "state_hash": completion.get("sha256", {}).get("recurrent_state") == sha256(job / "model/recurrent_state.pt"),
        "C4_route": c4.get("raw_history_visible_to_decoder") is False and c4.get("persistent_history_kv_cache") is False,
        "C4_target": c4.get("intervention_target") == "final_continuous_recurrent_state",
        "C4_support": int(summary.get("clean_n", 0)) >= 50,
        "C4_specificity": float(summary.get("D_specific", -1)) >= 0.15,
        "C4_noop": float((clean.get("noop") or {}).get("flip_to_A_rate", 1)) <= 0.05,
        "C4_same_rule": float((clean.get("samemal") or {}).get("flip_to_A_rate", 1)) <= 0.10,
        "C4_declared": summary.get("scientific_gate_pass") is True,
    }
    payload = {
        "stage": "M4_causal_rescue_full",
        "branch": branch,
        "pass": all(checks.values()),
        "checks": checks,
        "C4_summary": summary,
        "artifact_root": str(job),
    }
    if not payload["pass"]:
        raise ValueError(f"Causal-rescue full gate failed: {[k for k, v in checks.items() if not v]}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--branch",
        choices=(
            "raw4",
            "mean_normalized",
            "raw8",
            "mean2_normalized",
            "repeat100_w1",
            "repeat100_w4",
            "repeat100_sum_w1",
            "repeat100_sum_w4",
            "bootstrap_sum_w1",
            "bootstrap_sum_w4",
            "bootstrap_native25_sum_w1",
            "bootstrap_native25_sum_w4",
            "bootstrap_balanced_sum_w1",
            "bootstrap_balanced_sum_w4",
            "bootstrap_native25_balanced_sum_w1",
            "bootstrap_native25_balanced_sum_w4",
            "donor_directional_sum_w1",
            "donor_directional_sum_w4",
            "cf4_bootstrap_symmetric_w1",
            "cf4_donor_directional_w1",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(args.root, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
