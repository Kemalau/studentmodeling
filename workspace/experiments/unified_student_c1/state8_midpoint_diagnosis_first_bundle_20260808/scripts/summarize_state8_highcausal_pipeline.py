#!/usr/bin/env python3
"""Summarize the authoritative high-causal State8 training/evaluation chain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = (
    ("binding_screen", "state8_highcausal_binding_20260806"),
    ("unified_final_fit", "state8_highcausal_finalfit_20260806"),
    ("complete_final_eval", "state8_highcausal_complete_final_eval_20260806"),
)


def read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def heartbeat(path: Path, now: datetime) -> tuple[str | None, float | None]:
    value = read_text(path)
    if value is None:
        return None, None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat(), max(0.0, (now - parsed).total_seconds())


def summarize(results: Path, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    stages: dict[str, Any] = {}
    for name, directory in STAGES:
        root = results / directory
        stamp, age = heartbeat(root / "controller.heartbeat", now)
        passed = (root / "scientific_gate_pass").is_file()
        failed = (root / "scientific_gate_failed").is_file()
        complete = (root / "complete").is_file()
        if passed and failed:
            raise ValueError(f"Contradictory scientific gates: {root}")
        if passed and not complete:
            raise ValueError(f"Scientific pass without completion: {root}")
        stages[name] = {
            "directory": str(root),
            "active_experiment": read_text(root / "active_experiment.txt"),
            "heartbeat": stamp,
            "heartbeat_age_seconds": age,
            "controller_alive": age is not None and age <= 900,
            "complete": complete,
            "scientific_gate_pass": passed,
            "scientific_gate_failed": failed,
            "needs_attention": (root / "needs_attention").is_file(),
        }

    progress_root = results / "state8_highcausal_training_progress_20260806"
    progress = {
        arm: read_json(progress_root / arm / "progress.json")
        for arm in ("joint", "updater", "mid", "eedi", "taxonomy")
    }
    progress_attention = {
        arm: (progress_root / arm / "stalled.json").is_file()
        for arm in progress
    }
    selection = read_json(results / STAGES[0][1] / "selection.json")
    final_fit_receipt = read_json(
        results / STAGES[1][1] / "state8_final_fit_receipt.json"
    )
    canonical_receipt = read_json(
        results / STAGES[1][1] / "canonical_publish_receipt.json"
    )
    final_gate = read_json(results / STAGES[2][1] / "final_gate.json")
    ordered = [name for name, _ in STAGES]
    current = next((name for name in ordered if not stages[name]["complete"]), "complete")
    overall_complete = bool(
        final_gate
        and final_gate.get("stage") == "state8_complete_unified_model_final_gate_v1"
        and final_gate.get("pass") is True
        and stages["complete_final_eval"]["scientific_gate_pass"]
        and canonical_receipt
        and canonical_receipt.get("stage")
        == "state8_canonical_final_checkpoint_publish_v1"
        and canonical_receipt.get("status") == "complete"
    )
    return {
        "stage": "state8_highcausal_pipeline_live_status_v1",
        "observed_at": now.isoformat(),
        "current_stage": current,
        "overall_complete": overall_complete,
        "attention": [
            *[name for name in ordered if stages[name]["needs_attention"]],
            *[f"binding_progress_{arm}" for arm, value in progress_attention.items() if value],
        ],
        "stages": stages,
        "binding_progress": progress,
        "selection": selection,
        "final_fit_receipt": final_fit_receipt,
        "canonical_publish_receipt": canonical_receipt,
        "final_gate": final_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
