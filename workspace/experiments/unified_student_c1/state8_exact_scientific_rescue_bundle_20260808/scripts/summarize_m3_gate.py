#!/usr/bin/env python3
"""Summarize M3 generation-only versus behavioral-KD recurrent arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = ("generation_only", "generation_kd")


def arm_matches_path(arm: str, path: Path) -> bool:
    rendered = str(path).lower()
    return arm in rendered or arm.replace("_", "-") in rendered


def select_arm(rows: dict[str, dict]) -> tuple[str | None, str | None]:
    """Prefer a strict M3 pass, otherwise authorize a bounded M4 rescue.

    M3 is an intermediate bottleneck screen.  A non-collapsed arm that misses
    the strict recurrent/direct parity threshold may still be repaired by M4,
    which adds paired causal grounding and is followed by the full formal/C4
    gate.  No final fit is authorized merely by this rescue selection.
    """
    strict = [arm for arm in EXPECTED if rows[arm]["pass"]]
    rescue = [arm for arm in EXPECTED if rows[arm]["rescue_pass"]]
    eligible = strict or rescue
    if not eligible:
        return None, None
    selected = max(
        eligible,
        key=lambda arm: (
            rows[arm]["metrics"]["accuracy_gap_recurrent_minus_direct"],
            -rows[arm]["metrics"]["gold_nll_gap_recurrent_minus_direct"],
        ),
    )
    return selected, "strict_m3" if strict else "m4_causal_rescue"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--projection", choices=("identity", "trainable"), required=True)
    parser.add_argument("--max-overall-gap", type=float, default=0.10)
    parser.add_argument("--max-task-gap", type=float, default=0.15)
    parser.add_argument("--max-rescue-overall-gap", type=float, default=0.20)
    parser.add_argument("--max-rescue-task-gap", type=float, default=0.30)
    args = parser.parse_args()

    rows = {}
    for arm in EXPECTED:
        matches = [
            path
            for path in args.input_root.rglob("validation_metrics.json")
            if arm_matches_path(arm, path)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one validation file for {arm}, found {matches}")
        metrics = json.loads(matches[0].read_text(encoding="utf-8"))
        by_task = metrics.get("by_task") or {}
        task_checks = {
            task: values["accuracy_gap_recurrent_minus_direct"] >= -args.max_task_gap
            for task, values in by_task.items()
        }
        rescue_task_checks = {
            task: values["accuracy_gap_recurrent_minus_direct"]
            >= -args.max_rescue_task_gap
            for task, values in by_task.items()
        }
        checks = {
            "candidate_support": int(metrics.get("candidate_n", 0)) >= 100,
            "overall_gap": float(metrics["accuracy_gap_recurrent_minus_direct"])
            >= -args.max_overall_gap,
            "nonconstant_variance": float(metrics.get("state_variance", 0.0)) >= 1e-8,
            "nonconstant_cosine": float(metrics.get("mean_pairwise_cosine", 1.0)) <= 0.9999,
            "all_reported_tasks_within_gap": bool(task_checks) and all(task_checks.values()),
        }
        rows[arm] = {
            "source": str(matches[0]),
            "pass": all(checks.values()),
            "checks": checks,
            "task_checks": task_checks,
            "rescue_checks": {
                "candidate_support": int(metrics.get("candidate_n", 0)) >= 100,
                "direct_task_sufficiency": float(metrics.get("direct_accuracy", 0.0))
                >= 0.50,
                "bounded_overall_gap": float(
                    metrics["accuracy_gap_recurrent_minus_direct"]
                )
                >= -args.max_rescue_overall_gap,
                "nonconstant_variance": float(metrics.get("state_variance", 0.0))
                >= 1e-8,
                "nonconstant_cosine": float(metrics.get("mean_pairwise_cosine", 1.0))
                <= 0.9999,
                "all_reported_tasks_within_rescue_gap": bool(rescue_task_checks)
                and all(rescue_task_checks.values()),
            },
            "rescue_task_checks": rescue_task_checks,
            "metrics": metrics,
        }
        rows[arm]["rescue_pass"] = all(rows[arm]["rescue_checks"].values())
    selected, selection_mode = select_arm(rows)
    strict_pass = selection_mode == "strict_m3"
    rescue_authorized = selection_mode == "m4_causal_rescue"
    payload = {
        "stage": "M3_behavioral_distillation",
        "selected_projection_mode": args.projection,
        "scientific_gate_pass": strict_pass,
        "downstream_rescue_authorized": rescue_authorized,
        "selection_mode": selection_mode,
        "selected_arm": selected,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# M3 behavioral-distillation gate",
        "",
        "| arm | strict | rescue | recurrent acc | direct acc | gap | NLL gap | state cosine | variance | pairwise cosine | n |",
        "|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in rows.items():
        m = row["metrics"]
        lines.append(
            f"| {arm} | {row['pass']} | {row['rescue_pass']} | {m['recurrent_accuracy']:.3f} | "
            f"{m['direct_accuracy']:.3f} | {m['accuracy_gap_recurrent_minus_direct']:.3f} | "
            f"{m['gold_nll_gap_recurrent_minus_direct']:.3f} | {m['recurrent_direct_cosine']:.3f} | "
            f"{m['state_variance']:.3g} | {m['mean_pairwise_cosine']:.3f} | {m['candidate_n']} |"
        )
    lines.extend(
        ["", f"Selected arm: `{selected}`", f"Selection mode: `{selection_mode}`", ""]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
