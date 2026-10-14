#!/usr/bin/env python3
"""Gate the isolated pair-8 bootstrap before spending four GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-branch",
        choices=(
            "pair8_sum_bootstrap",
            "pair8_sum_bootstrap_native25",
            "pair8_sum_bootstrap_balanced",
            "pair8_sum_bootstrap_native25_balanced",
        ),
        default="pair8_sum_bootstrap",
    )
    args = parser.parse_args()
    paths = sorted(args.root.rglob("pair_overfit_summary.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one bootstrap summary, got {paths}")
    row = load(paths[0])
    if row.get("branch") != args.expected_branch:
        raise ValueError(f"Unexpected bootstrap branch: {row}")
    if row.get("preference_reduction") != "sum":
        raise ValueError(f"Bootstrap reduction drift: {row}")
    phases = row.get("objective_phase_counts") or {}
    objective_mode = row.get("objective_mode", "bootstrap_then_symmetric")
    expected_phases = (
        {"directional": 50, "symmetric": 150}
        if objective_mode == "bootstrap_then_symmetric"
        else {"directional_bootstrap": 50, "directional_balanced": 150}
    )
    first_margin = float(row["margin_loss_first_50"])
    last_margin = float(row["margin_loss_last_50"])
    first_delta = float(row["state_relative_delta_first_50"])
    last_delta = float(row["state_relative_delta_last_50"])
    validation = row.get("causal_validation") or {}
    native_accuracy = float(validation.get("native_accuracy", 0.0))
    donor_accuracy = float(validation.get("donor_accuracy", 0.0))
    directed_flip = float(validation.get("directed_flip_to_A_on_clean", 0.0))
    checks = {
        "curriculum_executed": phases == expected_phases,
        "margin_drop": first_margin - last_margin >= 0.05,
        "symmetric_tail_trainable": last_margin <= 0.45,
        "state_path_separated": last_delta >= max(first_delta * 1.05, 1e-5),
        "validation_support": int(validation.get("n", 0)) == 8,
        "clean_support": int(validation.get("clean_n", 0)) >= 4,
        "native_retained": native_accuracy >= 0.50,
        "donor_transfer": donor_accuracy >= 0.50,
        "directed_flip": directed_flip >= 0.50,
    }
    payload = {
        "stage": "M4_causal_pair8_bootstrap",
        "checks": checks,
        "pass": all(checks.values()),
        "margin_drop": first_margin - last_margin,
        "state_relative_delta_gain": last_delta - first_delta,
        "summary": row,
        "source": str(paths[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit("Bootstrap did not prove a trainable, separated causal state path")


if __name__ == "__main__":
    main()
