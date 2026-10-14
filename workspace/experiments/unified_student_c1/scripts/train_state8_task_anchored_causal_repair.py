#!/usr/bin/env python3
"""Taxonomy causal repair with task-conditional teacher-state anchoring.

A fixed trajectory weight is a poor fit for the unified objective: a weak
weight does not bind Eedi diagnosis semantics to the recurrent state, while a
strong weight on every task erodes T6.  This entrypoint keeps the qualified
frozen full-history teacher and strict state-only decoder, but applies a
strong, annealed trajectory target only to natural diagnosis rows.  Other
tasks retain a small anchor.  Swap, random-neutral, and same-rule objectives
are inherited unchanged from ``train_state8_taxonomy_causal_repair``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_taxonomy_causal_repair as taxonomy


def cli_value(name: str) -> str:
    if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
        raise ValueError(f"Missing required argument {name}")
    return sys.argv[sys.argv.index(name) + 1]


def parse_schedule(value: str) -> tuple[float, float, float]:
    pieces = tuple(float(piece.strip()) for piece in value.split(","))
    if len(pieces) != 3 or any(weight <= 0 for weight in pieces):
        raise ValueError(
            "STATE8_T3_ANCHOR_SCHEDULE must contain three positive weights"
        )
    if not (pieces[0] >= pieces[1] >= pieces[2]):
        raise ValueError("T3 anchor weights must be non-increasing")
    return pieces


def anchor_phase(progress: float) -> int:
    if not 0.0 <= float(progress) <= 1.0:
        raise ValueError("Training progress must be in [0, 1]")
    if progress < 0.40:
        return 0
    if progress < 0.70:
        return 1
    return 2


def is_diagnosis(row: dict[str, Any]) -> bool:
    return str(row.get("task")) == "diagnose_misconception"


def main() -> None:
    schedule = parse_schedule(
        os.environ.get("STATE8_T3_ANCHOR_SCHEDULE", "1.0,0.5,0.25")
    )
    other_weight = float(os.environ.get("STATE8_OTHER_ANCHOR_WEIGHT", "0.1"))
    configured_weight = float(
        os.environ.get("STATE8_REPAIR_TRAJECTORY_WEIGHT", "1.0")
    )
    if other_weight < 0:
        raise ValueError("STATE8_OTHER_ANCHOR_WEIGHT must be non-negative")
    if abs(configured_weight - max(schedule)) > 1e-12:
        raise ValueError(
            "Configured trajectory weight must equal the maximum T3 anchor"
        )

    output_dir = Path(cli_value("--output-dir")).resolve()
    original_parse = argparse.ArgumentParser.parse_args
    original_objective = trainer.backward_training_objective

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        namespace.task_anchor_schedule = list(schedule)
        namespace.task_anchor_boundaries = [0.40, 0.70]
        namespace.task_anchor_other_weight = other_weight
        namespace.task_anchor_scope = "diagnose_misconception"
        namespace.task_anchor_teacher = (
            "qualified_external_full_history_layerwise_state"
        )
        namespace.task_anchor_visible_at_inference = False
        return namespace

    def objective(
        wrapper,
        tokenizer,
        row,
        causal_row,
        cache,
        online_kd_cache,
        namespace,
        device,
        state_token_id,
        loss_scale,
        training_progress,
    ):
        phase = anchor_phase(float(training_progress))
        weight = schedule[phase] if is_diagnosis(row) else other_weight
        before = float(namespace.consistency_weight)
        namespace.consistency_weight = float(weight)
        try:
            record = original_objective(
                wrapper,
                tokenizer,
                row,
                causal_row,
                cache,
                online_kd_cache,
                namespace,
                device,
                state_token_id,
                loss_scale,
                training_progress,
            )
        finally:
            namespace.consistency_weight = before
        record["task_anchor_phase"] = phase
        record["task_anchor_effective_weight"] = float(weight)
        record["task_anchor_applied_to_diagnosis"] = is_diagnosis(row)
        return record

    argparse.ArgumentParser.parse_args = parse
    trainer.backward_training_objective = objective
    try:
        taxonomy.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            history = json.loads(
                (output_dir / "trainer_history.json").read_text(encoding="utf-8")
            )
            if not isinstance(history, list) or not history:
                raise ValueError("Task-anchor training history is empty")
            diagnosis_rows = [
                row for row in history if row.get("task") == "diagnose_misconception"
            ]
            other_rows = [
                row for row in history if row.get("task") != "diagnose_misconception"
            ]
            if not diagnosis_rows or not other_rows:
                raise ValueError("Task-anchor audit needs diagnosis and retention rows")
            phase_counts = Counter(int(row["task_anchor_phase"]) for row in history)
            if any(
                float(row["task_anchor_effective_weight"])
                != schedule[int(row["task_anchor_phase"])]
                for row in diagnosis_rows
            ):
                raise ValueError("A diagnosis row used the wrong phase anchor")
            if any(
                float(row["task_anchor_effective_weight"]) != other_weight
                for row in other_rows
            ):
                raise ValueError("A retention row used the wrong weak anchor")
            receipt = {
                "stage": "state8_task_anchored_taxonomy_causal_repair_v1",
                "status": "complete",
                "state_token_count": 8,
                "diagnosis_anchor_schedule": list(schedule),
                "schedule_boundaries": [0.40, 0.70],
                "other_task_anchor_weight": other_weight,
                "phase_counts": dict(sorted(phase_counts.items())),
                "diagnosis_rows": len(diagnosis_rows),
                "other_rows": len(other_rows),
                "teacher": "qualified_external_full_history_layerwise_state",
                "teacher_trainable": False,
                "raw_history_visible_to_decoder": False,
                "final_inference_route": "strict_state_only",
                "entrypoint_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
            (output_dir / "task_anchored_causal_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.backward_training_objective = original_objective


if __name__ == "__main__":
    main()
