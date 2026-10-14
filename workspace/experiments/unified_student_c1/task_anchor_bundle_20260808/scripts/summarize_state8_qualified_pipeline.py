#!/usr/bin/env python3
"""Summarize the qualified State8 pipeline from local controller evidence.

Remote job state is deliberately not inferred here.  Controllers write their
active experiment and terminal receipts; absent evidence remains ``pending``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = (
    ("teacher_curriculum", "state8_qualified_teacher_pipeline_20260805"),
    ("semantic_candidate_screen", "state8_semantic_training_progress_20260806"),
    ("balanced_curriculum", "state8_qualified_balanced_teacher_pipeline_20260805"),
    ("highkd_curriculum", "state8_qualified_highkd_teacher_pipeline_20260805"),
    ("primary_promotion", "state8_qualified_promotion_eval_20260805"),
    ("balanced_promotion", "state8_qualified_balanced_promotion_eval_20260805"),
    ("highkd_promotion", "state8_qualified_highkd_promotion_eval_20260805"),
    ("legacy_promotion_selection", "state8_qualified_promotion_selection_20260805"),
    ("unified_final_fit", "state8_unified_final_fit_20260805"),
    ("complete_final_eval", "state8_complete_final_eval_20260805"),
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


def heartbeat_age(path: Path, now: datetime) -> tuple[str | None, float | None]:
    value = read_text(path)
    if value is None:
        return None, None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat(), (now - parsed).total_seconds()


def summarize(results: Path, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    stages: dict[str, Any] = {}
    for name, directory_name in STAGES:
        root = results / directory_name
        controller_root = root
        if name == "teacher_curriculum":
            primary = root / "primary_curriculum"
            if (root / "teacher_gate_pass").is_file() and not (
                root / "complete"
            ).is_file():
                # Stage A publishes its qualified teacher into the root, then
                # the independent primary Stage-B owner trains the state8
                # student and finally publishes root/complete.  Reporting the
                # old Stage-A heartbeat here hid a dead Stage-B watcher.
                controller_root = primary
                heartbeat, age = heartbeat_age(
                    controller_root / "controller.heartbeat", now
                )
            else:
                accelerators = (
                    root / "sharded_clean_v2_16way",
                    root / "sharded_clean_v2",
                )
                live = []
                for candidate in accelerators:
                    if (candidate / "retired_in_favor_of_16way.txt").is_file():
                        continue
                    candidate_heartbeat, candidate_age = heartbeat_age(
                        candidate / "controller.heartbeat", now
                    )
                    if candidate_age is not None and candidate_age <= 900:
                        live.append(
                            (candidate, candidate_heartbeat, candidate_age)
                        )
                if live and not (root / "complete").is_file():
                    # Tuple order is intentional: the 16-way accelerator is
                    # preferred while both independently reproduce Stage A.
                    controller_root, heartbeat, age = live[0]
                else:
                    heartbeat, age = heartbeat_age(
                        root / "controller.heartbeat", now
                    )
        else:
            heartbeat, age = heartbeat_age(root / "controller.heartbeat", now)
        complete = (root / "complete").is_file()
        passed = (root / "scientific_gate_pass").is_file()
        failed = (root / "scientific_gate_failed").is_file()
        progress_root = controller_root / "progress"
        progress = read_json(progress_root / "progress.json")
        progress_stalled = (progress_root / "stalled.json").is_file()
        health_alert = (progress_root / "health_alert.json").is_file()
        attention = (root / "needs_attention").is_file() or (
            controller_root / "needs_attention"
        ).is_file() or progress_stalled or health_alert
        if passed and failed:
            raise ValueError(f"Contradictory scientific flags: {root}")
        if passed and not complete:
            raise ValueError(f"Scientific pass without completion: {root}")
        stages[name] = {
            "directory": str(root),
            "controller_directory": str(controller_root),
            "active_experiment": read_text(controller_root / "active_experiment.txt"),
            "heartbeat": heartbeat,
            "heartbeat_age_seconds": age,
            "controller_alive": age is not None and 0 <= age <= 900,
            "needs_attention": attention,
            "complete": complete,
            "scientific_gate_pass": passed,
            "scientific_gate_failed": failed,
            "training_progress": progress,
            "progress_stalled": progress_stalled,
            "health_alert": health_alert,
        }
    ordered = [name for name, _ in STAGES]
    critical = (
        "teacher_curriculum",
        "semantic_candidate_screen",
        "unified_final_fit",
        "complete_final_eval",
    )
    current = next((name for name in critical if not stages[name]["complete"]), "complete")
    final_gate = read_json(
        results / "state8_complete_final_eval_20260805/final_gate.json"
    )
    final_receipts = sorted(
        (results / "state8_unified_final_fit_20260805/download").glob(
            "**/state8_final_fit_receipt.json"
        )
    )
    if len(final_receipts) > 1:
        raise ValueError("Multiple final-fit receipts")
    final_receipt = read_json(final_receipts[0]) if final_receipts else None
    overall_complete = bool(
        final_gate
        and final_gate.get("stage") == "state8_complete_unified_model_final_gate_v1"
        and final_gate.get("pass") is True
        and stages["complete_final_eval"]["scientific_gate_pass"]
    )
    return {
        "stage": "state8_qualified_pipeline_live_status_v1",
        "observed_at": now.isoformat(),
        "current_stage": current,
        "overall_complete": overall_complete,
        "attention": [name for name in ordered if stages[name]["needs_attention"]],
        "stages": stages,
        "final_fit_receipt": final_receipt,
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
