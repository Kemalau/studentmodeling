#!/usr/bin/env python3
"""Diagnosis-first State8 repair with explicit T6 retention anchoring.

This conditional rescue is used only when the exact diagnosis curriculum
misses a non-diagnosis retention gate.  Diagnosis rows keep the annealed
full-history-teacher anchor, persistence rows receive an increasing anchor,
and the remaining unified tasks keep the weak background anchor.  The
decoder remains strictly state-only and inherits swap, random-neutral, and
same-rule causal objectives from the diagnosis-first repair.
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
import train_state8_diagnosis_first_causal_repair as diagnosis_first
from train_state8_task_anchored_causal_repair import (
    anchor_phase,
    cli_value,
    parse_schedule,
)


def task_kind(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "")
    if task == "diagnose_misconception":
        return "diagnosis"
    if task == "predict_error_persistence":
        return "persistence"
    return "other"


def effective_anchor_weight(
    row: dict[str, Any],
    progress: float,
    diagnosis_schedule: tuple[float, float, float],
    persistence_schedule: tuple[float, float, float],
    other_weight: float,
) -> tuple[int, str, float]:
    phase = anchor_phase(float(progress))
    kind = task_kind(row)
    if kind == "diagnosis":
        weight = diagnosis_schedule[phase]
    elif kind == "persistence":
        weight = persistence_schedule[phase]
    else:
        weight = float(other_weight)
    return phase, kind, float(weight)


def main() -> None:
    diagnosis_schedule = parse_schedule(
        os.environ.get("STATE8_T3_ANCHOR_SCHEDULE", "1.0,0.5,0.25")
    )
    persistence_schedule = tuple(
        float(piece.strip())
        for piece in os.environ.get(
            "STATE8_T6_ANCHOR_SCHEDULE", "0.5,0.75,1.0"
        ).split(",")
    )
    if len(persistence_schedule) != 3 or any(
        weight <= 0 for weight in persistence_schedule
    ):
        raise ValueError(
            "STATE8_T6_ANCHOR_SCHEDULE must contain three positive weights"
        )
    if not (
        persistence_schedule[0]
        <= persistence_schedule[1]
        <= persistence_schedule[2]
    ):
        raise ValueError("T6 anchor weights must be non-decreasing")
    other_weight = float(os.environ.get("STATE8_OTHER_ANCHOR_WEIGHT", "0.1"))
    configured_weight = float(
        os.environ.get("STATE8_REPAIR_TRAJECTORY_WEIGHT", "1.0")
    )
    if other_weight < 0:
        raise ValueError("STATE8_OTHER_ANCHOR_WEIGHT must be non-negative")
    if abs(
        configured_weight
        - max(max(diagnosis_schedule), max(persistence_schedule))
    ) > 1e-12:
        raise ValueError(
            "Configured trajectory weight must equal the maximum task anchor"
        )

    output_dir = Path(cli_value("--output-dir")).resolve()
    original_parse = argparse.ArgumentParser.parse_args
    original_objective = trainer.backward_training_objective

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        namespace.task_anchor_schedule = list(diagnosis_schedule)
        namespace.persistence_anchor_schedule = list(persistence_schedule)
        namespace.task_anchor_boundaries = [0.40, 0.70]
        namespace.task_anchor_other_weight = other_weight
        namespace.task_anchor_scope = [
            "diagnose_misconception",
            "predict_error_persistence",
        ]
        namespace.task_anchor_teacher = (
            "qualified_external_full_history_layerwise_state"
        )
        namespace.task_anchor_visible_at_inference = False
        namespace.diagnosis_first_retention_anchor_composed = True
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
        phase, kind, weight = effective_anchor_weight(
            row,
            float(training_progress),
            diagnosis_schedule,
            persistence_schedule,
            other_weight,
        )
        before = float(namespace.consistency_weight)
        namespace.consistency_weight = weight
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
        record["task_anchor_kind"] = kind
        record["task_anchor_effective_weight"] = weight
        record["diagnosis_first_retention_anchor_composed"] = True
        return record

    argparse.ArgumentParser.parse_args = parse
    trainer.backward_training_objective = objective
    try:
        diagnosis_first.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            history = json.loads(
                (output_dir / "trainer_history.json").read_text(encoding="utf-8")
            )
            diagnosis_receipt = json.loads(
                (output_dir / "diagnosis_first_causal_repair.json").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(history, list) or not history:
                raise ValueError("Retention-anchored training history is empty")
            kinds = Counter(str(row["task_anchor_kind"]) for row in history)
            if not all(kinds[kind] > 0 for kind in ("diagnosis", "persistence", "other")):
                raise ValueError(f"Retention anchor lacks task support: {kinds}")
            for row in history:
                phase = int(row["task_anchor_phase"])
                kind = str(row["task_anchor_kind"])
                expected = (
                    diagnosis_schedule[phase]
                    if kind == "diagnosis"
                    else persistence_schedule[phase]
                    if kind == "persistence"
                    else other_weight
                )
                if float(row["task_anchor_effective_weight"]) != float(expected):
                    raise ValueError("A row used the wrong task-conditional anchor")
            if diagnosis_receipt.get("field_order") != [
                "diagnosis",
                "evidence",
                "label",
            ]:
                raise ValueError("Diagnosis-first field order drift")
            receipt = {
                "stage": "state8_diagnosis_first_retention_anchored_causal_repair_v1",
                "status": "complete",
                "state_token_count": 8,
                "diagnosis_anchor_schedule": list(diagnosis_schedule),
                "persistence_anchor_schedule": list(persistence_schedule),
                "schedule_boundaries": [0.40, 0.70],
                "other_task_anchor_weight": other_weight,
                "task_kind_counts": dict(sorted(kinds.items())),
                "field_order": ["diagnosis", "evidence", "label"],
                "canonical_label_role": "auxiliary_ordinary_weight",
                "label_auxiliary_weight": 1.0,
                "teacher": "qualified_external_full_history_layerwise_state",
                "teacher_trainable": False,
                "raw_history_visible_to_decoder": False,
                "final_inference_route": "strict_state_only",
                "entrypoint_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
            (
                output_dir
                / "diagnosis_first_retention_anchored_causal_repair.json"
            ).write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.backward_training_objective = original_objective


if __name__ == "__main__":
    main()
