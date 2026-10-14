#!/usr/bin/env python3
"""Train T3 recurrence toward a label-informed, frozen semantic prototype.

The ordinary direct-state teacher captures the state carrier before answer
tokens, so it cannot encode information that the raw-history task adapter did
not already infer.  For Eedi misconception rows only, this diagnostic moves
the single carrier after the gold diagnosis while running a frozen external
adapter.  The recurrent student still receives only the observed interaction;
the label is used solely to construct a detached train-time trajectory target.

All non-Eedi rows retain the ordinary frozen external direct-state teacher.
Inference, state-only generation, causal losses, and the trainable student
adapter are unchanged.  This file is deliberately separate from the frozen
production trainer and from the already submitted external-teacher diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from open_t3_training_view import apply_open_t3_training_view
from train_recurrent_student_m4_external_state_teacher import (
    STUDENT_ADAPTER_NAME,
    TEACHER_ADAPTER_NAME,
    cli_path,
    external_teacher_spec,
    load_external_teacher,
)


TEACHER_SCOPE = "gold_semantic_eedi_t3_layerwise_trajectory"


def is_gold_semantic_row(row: dict[str, Any]) -> bool:
    return (
        row.get("task") == "diagnose_misconception"
        and row.get("source") == "eedi_mining"
    )


def gold_semantic_carrier_batch(
    input_ids: torch.Tensor,
    state_token_id: int,
    state_token_count: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the sole carrier after the supervised answer without adding tokens."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Gold-semantic teacher expects one input sequence")
    positions = torch.nonzero(
        input_ids[0].eq(int(state_token_id)), as_tuple=False
    ).flatten()
    if positions.numel() != int(state_token_count):
        raise ValueError(
            f"Gold-semantic teacher found {positions.numel()} state carriers"
        )
    start = int(positions[0].item())
    stop = int(positions[-1].item()) + 1
    if stop - start != int(state_token_count):
        raise ValueError("Gold-semantic state carriers must be contiguous")
    carrier = input_ids[:, start:stop]
    without = torch.cat([input_ids[:, :start], input_ids[:, stop:]], dim=1)
    semantic_ids = torch.cat([without, carrier], dim=1)
    semantic_attention = torch.ones(
        semantic_ids.shape,
        dtype=torch.long,
        device=semantic_ids.device,
    )
    if not bool(semantic_ids[0, -state_token_count:].eq(int(state_token_id)).all()):
        raise AssertionError("Semantic carriers were not placed after the answer")
    return semantic_ids, semantic_attention


def capture_gold_semantic_teacher(
    wrapper: LayerwiseRecurrentStudentModel,
    original_capture,
    *args,
    **kwargs,
):
    """Capture a detached label-informed state and restore the student adapter."""
    backbone = wrapper.backbone
    teacher_name = getattr(
        backbone, "_studentmodeling_external_state_teacher", None
    )
    student_name = getattr(backbone, "_studentmodeling_student_adapter", None)
    if teacher_name != TEACHER_ADAPTER_NAME or student_name != STUDENT_ADAPTER_NAME:
        raise ValueError("Gold-semantic external teacher is not attached")
    if backbone.active_adapter != student_name:
        raise ValueError("Gold-semantic capture did not start from the student adapter")
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("Gold-semantic capture requires input_ids")
    semantic_ids, semantic_attention = gold_semantic_carrier_batch(
        input_ids, wrapper.state_token_id, wrapper.state_token_count
    )
    was_training = backbone.training
    backbone.set_adapter(teacher_name, inference_mode=True)
    backbone.eval()
    try:
        with torch.no_grad():
            return original_capture(
                wrapper,
                input_ids=semantic_ids,
                attention_mask=semantic_attention,
                labels=None,
            )
    finally:
        backbone.set_adapter(student_name, inference_mode=False)
        backbone.train(was_training)


def main() -> None:
    teacher_path, teacher_digest = external_teacher_spec()
    train_path = cli_path("--train-jsonl")
    output_dir = cli_path("--output-dir")
    os.environ["STUDENT_TRAINING_VIEW"] = "open_t3_no_candidate_enumeration"

    original_parse_args = argparse.ArgumentParser.parse_args
    original_load = trainer.load_replicated_data_parallel_adapter
    original_read_jsonl = trainer.read_jsonl
    original_capture = LayerwiseRecurrentStudentModel.capture_direct_state
    original_objective = trainer.backward_training_objective
    training_view_audits: list[dict[str, object]] = []
    counts = {"gold_semantic_eedi_t3_rows": 0, "external_direct_teacher_rows": 0}

    def parse_args_with_teacher(parser, *args, **kwargs):
        namespace = original_parse_args(parser, *args, **kwargs)
        if getattr(namespace, "state_architecture", None) != "layerwise":
            raise ValueError("Gold-semantic teacher requires layerwise state")
        if float(getattr(namespace, "direct_replay_weight", -1.0)) != 0.0:
            raise ValueError("Gold-semantic teacher requires zero direct replay weight")
        if float(getattr(namespace, "consistency_weight", 0.0)) <= 0.0:
            raise ValueError("Gold-semantic teacher requires positive consistency weight")
        namespace.external_state_teacher_adapter = str(teacher_path)
        namespace.external_state_teacher_sha256 = teacher_digest
        namespace.external_state_teacher_adapter_name = TEACHER_ADAPTER_NAME
        namespace.external_state_teacher_scope = TEACHER_SCOPE
        namespace.external_state_teacher_trainable = False
        namespace.gold_semantic_teacher_sources = ["eedi_mining"]
        namespace.gold_semantic_teacher_task = "diagnose_misconception"
        namespace.gold_semantic_label_visible_at_inference = False
        namespace.training_view = "open_t3_no_candidate_enumeration"
        return namespace

    def load_student_and_teacher(model, adapter_source):
        backbone = original_load(model, adapter_source)
        load_external_teacher(backbone, teacher_path)
        return backbone

    def read_with_open_t3(path):
        rows = original_read_jsonl(path)
        if Path(path).resolve() == train_path:
            audit = apply_open_t3_training_view(rows)
            if audit["changed_malrule_t3_rows"]:
                training_view_audits.append({"path": str(path), **audit})
        return rows

    def capture(self, *args, **kwargs):
        if bool(getattr(self, "_studentmodeling_gold_semantic_active", False)):
            return capture_gold_semantic_teacher(
                self, original_capture, *args, **kwargs
            )
        from train_recurrent_student_m4_external_state_teacher import (
            capture_from_external_teacher,
        )

        return capture_from_external_teacher(
            self, original_capture, *args, **kwargs
        )

    def objective(wrapper, tokenizer, row, *args, **kwargs):
        semantic = is_gold_semantic_row(row)
        key = (
            "gold_semantic_eedi_t3_rows"
            if semantic
            else "external_direct_teacher_rows"
        )
        counts[key] += 1
        wrapper._studentmodeling_gold_semantic_active = semantic
        try:
            return original_objective(wrapper, tokenizer, row, *args, **kwargs)
        finally:
            wrapper._studentmodeling_gold_semantic_active = False

    argparse.ArgumentParser.parse_args = parse_args_with_teacher
    trainer.load_replicated_data_parallel_adapter = load_student_and_teacher
    trainer.read_jsonl = read_with_open_t3
    trainer.backward_training_objective = objective
    LayerwiseRecurrentStudentModel.capture_direct_state = capture
    try:
        trainer.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if not training_view_audits:
                raise AssertionError("Gold-semantic run changed no Malrule T3 rows")
            if counts["gold_semantic_eedi_t3_rows"] <= 0:
                raise AssertionError("Gold-semantic run trained no Eedi T3 rows")
            receipt = {
                "stage": "gold_semantic_state_teacher_training_view_v1",
                "training_view": "open_t3_no_candidate_enumeration",
                "teacher_adapter": str(teacher_path),
                "teacher_adapter_sha256": teacher_digest,
                "teacher_scope": TEACHER_SCOPE,
                "teacher_trainable": False,
                "student_observes_gold_label": False,
                "gold_label_visible_at_inference": False,
                **counts,
                **training_view_audits[0],
            }
            (output_dir / "gold_semantic_state_teacher_training_view.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
        trainer.load_replicated_data_parallel_adapter = original_load
        trainer.read_jsonl = original_read_jsonl
        trainer.backward_training_objective = original_objective
        LayerwiseRecurrentStudentModel.capture_direct_state = original_capture


if __name__ == "__main__":
    main()
