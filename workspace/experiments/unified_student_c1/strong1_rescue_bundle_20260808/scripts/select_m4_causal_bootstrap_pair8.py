#!/usr/bin/env python3
"""Select the pair-8 curriculum that preserves native behavior and transfers donor state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_BRANCHES = {
    "donor_only": "pair8_sum_bootstrap",
    "native25": "pair8_sum_bootstrap_native25",
    "balanced": "pair8_sum_bootstrap_balanced",
    "native25_balanced": "pair8_sum_bootstrap_native25_balanced",
}


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage") != "M4_causal_pair8_bootstrap":
        raise ValueError(f"Invalid pair-8 gate: {path}")
    return value


def select(candidates: list[tuple[str, Path]]) -> dict:
    rows = []
    for variant, path in candidates:
        if variant not in EXPECTED_BRANCHES:
            raise ValueError(f"Unknown pair-8 variant: {variant}")
        if not path.is_file():
            rows.append(
                {
                    "variant": variant,
                    "gate": str(path),
                    "pass": False,
                    "failure": "missing_gate",
                    "balanced_accuracy": 0.0,
                    "native_accuracy": 0.0,
                    "donor_accuracy": 0.0,
                    "directed_flip": 0.0,
                    "clean_n": 0,
                    "margin_drop": 0.0,
                    "state_relative_delta_gain": 0.0,
                }
            )
            continue
        gate = load(path)
        summary = gate.get("summary") or {}
        if summary.get("branch") != EXPECTED_BRANCHES[variant]:
            raise ValueError(f"Pair-8 branch/variant drift: {variant} {summary}")
        validation = summary.get("causal_validation") or {}
        row = {
            "variant": variant,
            "gate": str(path),
            "pass": gate.get("pass") is True and all((gate.get("checks") or {}).values()),
            "native_accuracy": float(validation.get("native_accuracy", 0.0)),
            "donor_accuracy": float(validation.get("donor_accuracy", 0.0)),
            "directed_flip": float(validation.get("directed_flip_to_A_on_clean", 0.0)),
            "clean_n": int(validation.get("clean_n", 0)),
            "margin_drop": float(gate.get("margin_drop", 0.0)),
            "state_relative_delta_gain": float(gate.get("state_relative_delta_gain", 0.0)),
        }
        row["balanced_accuracy"] = min(
            row["native_accuracy"], row["donor_accuracy"]
        )
        rows.append(row)
    passing = [row for row in rows if row["pass"]]
    selected = max(
        passing,
        key=lambda row: (
            row["balanced_accuracy"],
            row["native_accuracy"],
            row["donor_accuracy"],
            row["directed_flip"],
            row["margin_drop"],
            row["state_relative_delta_gain"],
        ),
        default=None,
    )
    return {
        "stage": "M4_causal_pair8_curriculum_selection",
        "pass": selected is not None,
        "selected_variant": selected["variant"] if selected else None,
        "selected": selected,
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as VARIANT=/path/to/gate.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for raw in args.candidate:
        variant, separator, path = raw.partition("=")
        if not separator or not variant or not path:
            raise ValueError(f"Invalid candidate: {raw}")
        candidates.append((variant, Path(path)))
    payload = select(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit("No pair-8 curriculum passed the causal/native-retention gate")


if __name__ == "__main__":
    main()
