#!/usr/bin/env python3
"""Diagnosis-first State8 repair with task-conditional teacher anchoring.

This composes the two independently audited repairs without changing the
inference path.  Natural-language diagnosis and evidence remain before the
auxiliary canonical label, while only diagnosis rows receive the strong,
annealed full-history-teacher state anchor.  Other unified tasks retain a weak
anchor so T1/T2/T5/T6 are not overwritten by a diagnosis-only continuation.
The final decoder still receives only eight state tokens and the current query.
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
    is_diagnosis,
    parse_schedule,
)


def effective_anchor_weight(
    row: dict[str, Any], progress: float, schedule: tuple[float, float, float],
    other_weight: float,
) -> tuple[int, float]:
    phase = anchor_phase(float(progress))
    return phase, schedule[phase] if is_diagnosis(row) else float(other_weight)


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
        namespace.diagnosis_first_task_anchor_composed = True
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
        phase, weight = effective_anchor_weight(
            row, float(training_progress), schedule, other_weight
        )
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
        record["diagnosis_first_task_anchor_composed"] = True
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
                raise ValueError("Composed repair training history is empty")
            diagnosis_rows = [row for row in history if is_diagnosis(row)]
            other_rows = [row for row in history if not is_diagnosis(row)]
            if not diagnosis_rows or not other_rows:
                raise ValueError(
                    "Composed repair audit needs diagnosis and retention rows"
                )
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
            if diagnosis_receipt.get("field_order") != [
                "diagnosis", "evidence", "label"
            ]:
                raise ValueError("Diagnosis-first field order drift")
            receipt = {
                "stage": "state8_diagnosis_first_task_anchored_causal_repair_v1",
                "status": "complete",
                "state_token_count": 8,
                "diagnosis_anchor_schedule": list(schedule),
                "schedule_boundaries": [0.40, 0.70],
                "other_task_anchor_weight": other_weight,
                "phase_counts": dict(sorted(phase_counts.items())),
                "diagnosis_rows": len(diagnosis_rows),
                "other_rows": len(other_rows),
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
            (output_dir / "diagnosis_first_task_anchored_causal_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.backward_training_objective = original_objective


if __name__ == "__main__":
    main()
