#!/usr/bin/env python3
"""Compare the controlled eight-pair causal-path overfit diagnostics."""

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
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("pair_overfit_summary.json")):
        row = load(path)
        if row.get("branch") not in {
            "pair8_mean",
            "pair8_sum",
            "pair8_sum_bootstrap",
        }:
            raise ValueError(f"Unexpected pair-overfit branch: {row}")
        validation = row.get("causal_validation") or {}
        rows.append(
            {
                **row,
                "margin_drop": float(row["margin_loss_first_50"])
                - float(row["margin_loss_last_50"]),
                "directed_flip": float(validation.get("directed_flip_to_A_on_clean", 0.0)),
                "clean_n": int(validation.get("clean_n", 0)),
            }
        )
    required = {"pair8_mean", "pair8_sum"}
    if not required.issubset({row["branch"] for row in rows}):
        raise ValueError(f"Expected symmetric mean and sum branches, got {rows}")
    by_branch = {row["branch"]: row for row in rows}
    mean = by_branch["pair8_mean"]
    summed = by_branch["pair8_sum"]
    path_trainable = min(float(row["margin_loss_last_50"]) for row in rows) < 0.45
    sum_better = (
        float(summed["margin_loss_last_50"]) + 0.02
        < float(mean["margin_loss_last_50"])
        or float(summed["directed_flip"]) > float(mean["directed_flip"])
    )
    payload = {
        "stage": "M4_causal_pair8_overfit",
        "path_trainable": path_trainable,
        "sum_alignment_supported": sum_better,
        "bootstrap_supported": (
            "pair8_sum_bootstrap" in by_branch
            and (
                float(by_branch["pair8_sum_bootstrap"]["margin_loss_last_50"]) + 0.02
                < float(summed["margin_loss_last_50"])
                or float(by_branch["pair8_sum_bootstrap"]["directed_flip"])
                > float(summed["directed_flip"])
            )
        ),
        "branches": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not path_trainable:
        raise SystemExit("Causal state path did not overfit even eight repeated pairs")


if __name__ == "__main__":
    main()
