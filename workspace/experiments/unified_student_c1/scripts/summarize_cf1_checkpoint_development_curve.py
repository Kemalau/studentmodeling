#!/usr/bin/env python3
"""Summarize task, probe, and causal metrics over frozen CF1 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_stage_a_gate import TASK, clean_swap_summary


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_history_value(history: list[dict[str, Any]], step: int, key: str) -> Any:
    candidates = [row for row in history if int(row.get("step", -1)) <= step and key in row]
    if not candidates:
        return None
    return max(candidates, key=lambda row: int(row.get("step", -1)))[key]


def checkpoint_row(curve_dir: Path, run_dir: Path, step: int) -> dict[str, Any]:
    step_dir = curve_dir / f"step_{step}"
    response = read_json(step_dir / "response_cross_masked.json")
    probe = read_json(step_dir / "state_probe_masked.json")
    swap_path = step_dir / "state_swap_masked.json"
    swap_summary = clean_swap_summary(swap_path)

    state_layers = probe["metrics"]["state"]
    best_layer, best_values = max(
        state_layers.items(),
        key=lambda item: (
            float(item[1]["cross"]["accuracy"]),
            float(item[1]["cross"]["macro_f1"]),
        ),
    )
    clean = swap_summary["clean"]
    swap_rate = float(clean["swap"]["flip_to_A_rate"])
    random_rate = float(clean["random"]["flip_to_A_rate"])
    same_rule_rate = float(clean["samemal"]["flip_to_A_rate"])
    noop_rate = float(clean["noop"]["flip_to_A_rate"])

    checkpoint_dir = run_dir / f"checkpoint-{step}"
    trainer_state = read_json(checkpoint_dir / "trainer_state.json")
    history = trainer_state.get("log_history", [])
    return {
        "step": step,
        "epoch": trainer_state.get("epoch"),
        "train_loss_latest": latest_history_value(history, step, "loss"),
        "eval_loss_latest": latest_history_value(history, step, "eval_loss"),
        "response_cross_accuracy": float(response["model"]["metrics"][TASK]["accuracy"]),
        "response_cross_macro_f1": float(response["model"]["metrics"][TASK]["macro_f1"]),
        "state_probe_best_cross_accuracy": float(best_values["cross"]["accuracy"]),
        "state_probe_best_cross_macro_f1": float(best_values["cross"]["macro_f1"]),
        "state_probe_best_layer": int(best_layer),
        "clean_n": int(swap_summary["clean_n"]),
        "swap_flip_to_A": swap_rate,
        "random_flip_to_A": random_rate,
        "same_rule_flip_to_A": same_rule_rate,
        "noop_flip_to_A": noop_rate,
        "D_specific": swap_rate - random_rate,
        "D_damage": random_rate - noop_rate,
        "swap_changed_top1": float(clean["swap"]["changed_top1_rate"]),
        "swap_delta_margin": float(clean["swap"]["mean_delta_margin_A_minus_B"]),
        "checkpoint_adapter_sha256": (step_dir / "checkpoint_adapter.sha256").read_text(
            encoding="utf-8"
        ).strip(),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CF1 + vision-pad checkpoint development curve",
        "",
        "| step | train loss | cross acc | probe acc | probe layer | clean n | swap | random | same-rule | no-op | D_specific |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {step} | {loss} | {cross} | {probe} | {layer} | {clean_n} | {swap} | {random} | {same} | {noop} | {specific} |".format(
                step=row["step"],
                loss=fmt(row["train_loss_latest"]),
                cross=fmt(row["response_cross_accuracy"]),
                probe=fmt(row["state_probe_best_cross_accuracy"]),
                layer=row["state_probe_best_layer"],
                clean_n=row["clean_n"],
                swap=fmt(row["swap_flip_to_A"]),
                random=fmt(row["random_flip_to_A"]),
                same=fmt(row["same_rule_flip_to_A"]),
                noop=fmt(row["noop_flip_to_A"]),
                specific=fmt(row["D_specific"]),
            )
        )
    lines.extend(
        [
            "",
            "All checkpoints use the frozen cross-template split and frozen 150-pair manifest. "
            "`D_specific = swap - random`; clean pairs satisfy `none.category == stay_B` independently at each checkpoint.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", default="250,500,750,1000,1250,1500")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    steps = [int(value) for value in args.steps.split(",") if value]
    rows = [checkpoint_row(args.curve_dir, args.run_dir, step) for step in steps]
    payload = {
        "protocol": "cf1_visionpad_strict_checkpoint_development_v1",
        "steps": steps,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(rows), encoding="utf-8")
    print(args.output_md.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
