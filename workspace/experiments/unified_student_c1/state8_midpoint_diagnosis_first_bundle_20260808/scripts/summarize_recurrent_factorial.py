#!/usr/bin/env python3
"""Aggregate the four strict M2 jobs and select a projection mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "recurrent-u1-identity-seed42": (1, "identity"),
    "recurrent-u1-trainable-seed42": (1, "trainable"),
    "recurrent-u4-identity-seed42": (4, "identity"),
    "recurrent-u4-trainable-seed42": (4, "trainable"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for job, (unroll, projection) in EXPECTED.items():
        matches = list(args.input_root.glob(f"**/{job}/**/m2_gate.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected one m2_gate.json for {job}, found {matches}")
        gate = json.loads(matches[0].read_text(encoding="utf-8"))
        if int(gate["unroll"]) != unroll or gate["projection_mode"] != projection:
            raise ValueError(f"Gate metadata mismatch for {job}")
        metrics = gate["metrics"]
        rows.append(
            {
                "job": job,
                "unroll": unroll,
                "projection_mode": projection,
                "pass": bool(gate["scientific_gate_pass"]),
                "checks": gate["checks"],
                "recurrent_accuracy": metrics["recurrent_accuracy"],
                "direct_accuracy": metrics["direct_accuracy"],
                "accuracy_gap": metrics["accuracy_gap_recurrent_minus_direct"],
                "gold_nll_gap": metrics["gold_nll_gap_recurrent_minus_direct"],
                "state_cosine": metrics["recurrent_direct_cosine"],
                "state_variance": metrics["state_variance"],
                "mean_pairwise_cosine": metrics["mean_pairwise_cosine"],
                "candidate_n": metrics["candidate_n"],
                "path": str(matches[0]),
            }
        )
    eligible = []
    for projection in ("identity", "trainable"):
        pair = [row for row in rows if row["projection_mode"] == projection]
        if len(pair) == 2 and all(row["pass"] for row in pair):
            eligible.append(
                (
                    sum(row["accuracy_gap"] for row in pair) / 2,
                    -sum(row["gold_nll_gap"] for row in pair) / 2,
                    projection,
                )
            )
    strict_selected = max(eligible)[2] if eligible else None

    # M2 is an architecture/initialization diagnostic.  If the strict
    # near-parity criterion is not yet met, M3 behavioral distillation is the
    # explicit rescue stage.  Authorize that stage only from an unroll-4 arm
    # with adequate support, a learnable direct task, and a non-collapsed state;
    # never relabel the failed strict M2 result as a pass.
    rescue_candidates = []
    for row in rows:
        checks = row["checks"]
        if row["unroll"] != 4:
            continue
        core_checks = (
            "candidate_support",
            "direct_task_sufficiency",
            "nonconstant_variance",
            "nonconstant_cosine",
        )
        if all(bool(checks.get(name)) for name in core_checks):
            rescue_candidates.append(
                (
                    row["accuracy_gap"],
                    -row["gold_nll_gap"],
                    row["projection_mode"],
                )
            )
    rescue_selected = max(rescue_candidates)[2] if rescue_candidates else None
    selected = strict_selected or rescue_selected
    selection_mode = (
        "strict_m2_pass"
        if strict_selected is not None
        else "m3_behavioral_rescue"
        if rescue_selected is not None
        else None
    )
    payload = {
        "stage": "M2_recurrent_factorial",
        "all_four_present": len(rows) == 4,
        "selected_projection_mode": selected,
        "scientific_gate_pass": strict_selected is not None,
        "downstream_rescue_authorized": selected is not None,
        "selection_mode": selection_mode,
        "selection_rule": "both u1/u4 pass; maximize mean accuracy gap then minimize NLL gap",
        "rescue_rule": "if strict M2 fails, choose the best non-collapsed supported u4 arm for M3 behavioral distillation",
        "rows": sorted(rows, key=lambda row: (row["unroll"], row["projection_mode"])),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Strict recurrent-state M2 factorial",
        "",
        "| unroll | projection | pass | recurrent acc | direct acc | gap | gold NLL gap | state cosine | variance | pairwise cosine | n |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['unroll']} | {row['projection_mode']} | {row['pass']} | "
            f"{row['recurrent_accuracy']:.3f} | {row['direct_accuracy']:.3f} | "
            f"{row['accuracy_gap']:.3f} | {row['gold_nll_gap']:.3f} | "
            f"{row['state_cosine']:.3f} | {row['state_variance']:.3g} | "
            f"{row['mean_pairwise_cosine']:.3f} | {row['candidate_n']} |"
        )
    lines.extend(
        [
            "",
            f"Strict M2 pass: `{payload['scientific_gate_pass']}`",
            f"Downstream rescue authorized: `{payload['downstream_rescue_authorized']}`",
            f"Selection mode: `{selection_mode}`",
            f"Selected projection: `{selected}`",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
