#!/usr/bin/env python3
"""Summarize the active State8 repair-to-final-evaluation chain fail closed."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return value or None


def alive(path: Path) -> bool:
    value = read_text(path)
    if value is None or not value.isdigit():
        return False
    try:
        os.kill(int(value), 0)
    except OSError:
        return False
    return True


def marker_state(root: Path) -> dict[str, Any]:
    selection = read_json(root / "selection.json")
    return {
        "controller_alive": alive(root / "controller.active.pid"),
        "complete": (root / "complete").is_file(),
        "needs_attention": (root / "needs_attention").is_file(),
        "scientific_gate_pass": (root / "scientific_gate_pass").is_file(),
        "scientific_gate_failed": (root / "scientific_gate_failed").is_file(),
        "selection_pass": selection.get("pass") if isinstance(selection, dict) else None,
    }


def screen_state(root: Path) -> dict[str, Any]:
    latest = read_json(root / "latest_status.json")
    remote: str | None = None
    if isinstance(latest, list) and len(latest) == 1 and isinstance(latest[0], dict):
        remote = str(latest[0].get("status") or "").casefold() or None
    return {
        "complete": (root / "complete").is_file(),
        "needs_attention": (root / "needs_attention").is_file(),
        "summary_ready": (root / "summary.json").is_file(),
        "last_known_remote_status": remote,
    }


def summarize(project: Path, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    results = project / "results"
    screens = results / "state8_curriculum_repair_screens_20260812"
    repair_screens: dict[str, Any] = {}
    for seed, branch in (
        (8201, "causal_teacher_low"),
        (8202, "causal_teacher_high"),
        (8301, "early_label_low"),
        (8302, "early_label_high"),
    ):
        for step in (20, 100, 240, 500):
            name = f"seed{seed}_{branch}_cp{step}"
            repair_screens[name] = screen_state(
                screens / f"seed{seed}_{branch}/checkpoint-{step}"
            )

    interleaved_remote = results / "state8_interleaved_retention_recovery3_20260813"
    interleaved_status = read_text(interleaved_remote / "latest_status.tsv")
    selectors = {
        "causal_teacher": marker_state(
            results / "state8_causal_teacher_checkpoint_selection_20260812"
        ),
        "early_label": marker_state(
            results / "state8_early_label_checkpoint_selection_20260813"
        ),
        "interleaved": marker_state(
            results / "state8_interleaved_retention_selection_20260813"
        ),
        "global": marker_state(
            results / "state8_curriculum_global_selection_20260812"
        ),
    }
    promotion = marker_state(results / "state8_curriculum_promotion_20260812")
    final_fit = marker_state(results / "state8_curriculum_finalfit_world2_20260812")
    final_eval = marker_state(
        results / "state8_curriculum_complete_final_eval_20260812"
    )
    objective = marker_state(
        results / "state8_curriculum_complete_objective_20260812"
    )
    final_objective = read_json(
        results / "state8_curriculum_complete_objective_20260812/final_objective.json"
    )
    c3 = read_json(results / "c3_exposure_matched_world2_20260809/summary.json")
    flow_complete = objective["complete"] and isinstance(final_objective, dict)
    scientific_pass = bool(
        flow_complete
        and objective["scientific_gate_pass"]
        and final_objective.get("scientific_gate_pass") is True
    )
    attention = [
        name
        for name, value in {
            **{f"selector:{key}": value for key, value in selectors.items()},
            "promotion": promotion,
            "final_fit": final_fit,
            "final_eval": final_eval,
            "objective": objective,
        }.items()
        if value["needs_attention"]
    ]
    return {
        "stage": "state8_large_evaluation_live_status_v1",
        "observed_at": now.isoformat(),
        "feeds_monitoring_ready": not (
            results / "recurrent_unified_supervisor_20260731/auth_required"
        ).is_file(),
        "base_curriculum": marker_state(
            results / "state8_curriculum_selection_20260812"
        ),
        "repair_screens": repair_screens,
        "interleaved_remote": {
            "status": interleaved_status.split("\t")[-1] if interleaved_status else None,
            "remote_passed": (interleaved_remote / "remote_passed").is_file(),
            "remote_failed": (interleaved_remote / "remote_failed").is_file(),
        },
        "selectors": selectors,
        "promotion": promotion,
        "final_fit": final_fit,
        "final_eval": final_eval,
        "objective": objective,
        "c3": {
            "complete": (results / "c3_exposure_matched_world2_20260809/complete").is_file(),
            "scientific_gate_pass": bool(
                isinstance(c3, dict) and c3.get("scientific_gate_pass") is True
            ),
            "claim_supported": bool(
                isinstance(c3, dict)
                and (c3.get("conclusion") or {}).get("C3_transfer_hypothesis_supported") is True
            ),
        },
        "needs_attention": sorted(attention),
        "evaluation_flow_complete": flow_complete,
        "scientific_gate_pass": scientific_pass,
        "final_objective": final_objective,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
