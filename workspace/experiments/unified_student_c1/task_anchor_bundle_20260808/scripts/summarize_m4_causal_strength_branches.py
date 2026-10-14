#!/usr/bin/env python3
"""Compare controlled M4 causal-strength branches at round 500."""

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
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--expected-branches", nargs="+", default=["raw4", "mean_normalized"]
    )
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.root.rglob("branch_training_summary.json")):
        branch = load(summary_path)
        job_root = summary_path.parents[1]
        c4_paths = sorted(job_root.rglob("recurrent_causal_gate.json"))
        validation_paths = sorted(job_root.rglob("validation_metrics.json"))
        causal_validation_paths = sorted(job_root.rglob("causal_validation_metrics.json"))
        if len(c4_paths) != 1 or len(validation_paths) != 1 or len(causal_validation_paths) != 1:
            raise ValueError(f"Incomplete branch result under {job_root}")
        c4 = load(c4_paths[0])
        validation = load(validation_paths[0])
        causal_validation = load(causal_validation_paths[0])
        c4_summary = c4["summary"]
        clean = c4_summary["clean"]
        rows.append(
            {
                **branch,
                "C4_D_specific": float(c4_summary["D_specific"]),
                "C4_D_damage": float(c4_summary["D_damage"]),
                "C4_clean_n": int(c4_summary["clean_n"]),
                "C4_swap_flip": float(clean["swap"]["flip_to_A_rate"]),
                "C4_random_flip": float(clean["random"]["flip_to_A_rate"]),
                "C4_noop_flip": float(clean["noop"]["flip_to_A_rate"]),
                "C4_same_rule_flip": float(clean["samemal"]["flip_to_A_rate"]),
                "validation_recurrent_accuracy": float(validation["recurrent_accuracy"]),
                "validation_direct_accuracy": float(validation["direct_accuracy"]),
                "validation_accuracy_gap": float(validation["accuracy_gap_recurrent_minus_direct"]),
                "causal_native_accuracy": float(causal_validation["native_accuracy"]),
                "causal_donor_accuracy": float(causal_validation["donor_accuracy"]),
                "causal_directed_flip": float(causal_validation["directed_flip_to_A_on_clean"]),
            }
        )
    expected_branches = set(args.expected_branches)
    if {row["branch"] for row in rows} != expected_branches:
        raise ValueError(f"Expected both branches, got {[row['branch'] for row in rows]}")
    eligible = [
        row for row in rows
        if row["C4_clean_n"] >= 50
        and row["C4_noop_flip"] <= 0.05
        and row["C4_same_rule_flip"] <= 0.10
        and row["validation_accuracy_gap"] >= -0.05
    ]
    selected = max(eligible, key=lambda row: row["C4_D_specific"], default=None)
    payload = {
        "stage": "M4_causal_strength_round500",
        "branches": rows,
        "selected_branch": selected["branch"] if selected and selected["C4_D_specific"] >= 0.15 else None,
        "scientific_gate_pass": bool(selected and selected["C4_D_specific"] >= 0.15),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# M4 causal-strength round-500 comparison",
        "",
        "| branch | effective weight | margin first | margin second | C4 D_specific | C4 D_damage | clean n | recurrent acc | direct acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['branch']} | {row['effective_causal_weight']:.3f} | "
            f"{row['margin_loss_first_half']:.3f} | {row['margin_loss_second_half']:.3f} | "
            f"{row['C4_D_specific']:.3f} | {row['C4_D_damage']:.3f} | "
            f"{row['C4_clean_n']} | {row['validation_recurrent_accuracy']:.3f} | "
            f"{row['validation_direct_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Selected branch: `{payload['selected_branch']}`",
            f"Scientific gate pass: `{payload['scientific_gate_pass']}`",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
