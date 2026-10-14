#!/usr/bin/env python3
"""Five-stage State8 curriculum with non-vanishing endpoint teacher KD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from state8_natural_curriculum_view import apply_state8_natural_curriculum_view
from train_recurrent_student_m4_external_state_teacher import (
    TEACHER_ADAPTER_NAME,
    capture_from_external_teacher,
    cli_path,
    external_teacher_spec,
    load_external_teacher,
)
from train_state8_stateonly_teacher_repair import cli_value, schema_weighted_loss_factory


def repaired_dual_path_weights(args, progress: float) -> tuple[float, float, float]:
    """Preserve the five phases while keeping KD through state-only closure."""
    dropout = trainer.history_dropout_at_progress(args, progress)
    return 1.0 - dropout, dropout, 1.0 if dropout > 0.0 else 0.0


def main() -> None:
    teacher_path, teacher_digest = external_teacher_spec()
    train_path = cli_path("--train-jsonl")
    valid_path = cli_path("--valid-jsonl")
    output_dir = cli_path("--output-dir")
    trajectory_weight = float(os.environ.get("STATE8_REPAIR_TRAJECTORY_WEIGHT", "0"))
    label_weight = float(os.environ.get("STATE8_REPAIR_LABEL_WEIGHT", "4"))
    if trajectory_weight < 0:
        raise ValueError("Trajectory weight must be non-negative")
    tokenizer = AutoTokenizer.from_pretrained(cli_value("--model"), trust_remote_code=True)
    weighted_loss = schema_weighted_loss_factory(tokenizer, label_weight)

    original_parse = argparse.ArgumentParser.parse_args
    original_load = trainer.load_replicated_data_parallel_adapter
    original_read = trainer.read_jsonl
    original_capture = LayerwiseRecurrentStudentModel.capture_direct_state
    original_dual_weights = trainer.dual_path_curriculum_weights
    original_lm_loss = trainer.causal_lm_loss
    audits: list[dict[str, int | str]] = []

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        if namespace.state_architecture != "layerwise" or namespace.state_token_count != 8:
            raise ValueError("Repaired curriculum requires eight-slot layerwise state")
        if not namespace.history_dropout_curriculum or not namespace.dual_path_curriculum:
            raise ValueError("Repaired curriculum requires five-stage dual-path flags")
        if namespace.online_kd_weight <= 0:
            raise ValueError("Repaired curriculum requires positive teacher KD")
        if namespace.behavior_kd_weight != 0.0 or namespace.direct_replay_weight != 0.0:
            raise ValueError("Repaired curriculum forbids cached KD and direct replay")
        if abs(float(namespace.consistency_weight) - trajectory_weight) > 1e-12:
            raise ValueError("Trajectory-weight environment/config mismatch")
        namespace.external_state_teacher_adapter = str(teacher_path)
        namespace.external_state_teacher_sha256 = teacher_digest
        namespace.external_state_teacher_adapter_name = TEACHER_ADAPTER_NAME
        namespace.external_state_teacher_scope = (
            "answer_logits_and_direct_trajectory"
            if trajectory_weight > 0
            else "answer_logits_only"
        )
        namespace.external_state_teacher_trainable = False
        namespace.training_view = "state8_natural_label_diagnosis_evidence"
        namespace.output_schema = "label_diagnosis_evidence_v1"
        namespace.curriculum_route = "five_stage_nonvanishing_endpoint_teacher_kd"
        namespace.history_dropout_schedule = [0.0, 0.25, 0.5, 0.75, 1.0]
        namespace.online_teacher_scope = "qualified_external_full_history_answer_logits"
        namespace.trajectory_teacher_scope = (
            "qualified_external_direct_layerwise_trajectory"
            if trajectory_weight > 0
            else "none"
        )
        namespace.teacher_gate_required = True
        namespace.teacher_trainable = False
        namespace.final_inference_route = "strict_state_only"
        namespace.label_auxiliary_weight = label_weight
        namespace.endpoint_online_kd_phase_weight = 1.0
        namespace.entrypoint_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        namespace.state8_natural_view_sha256 = hashlib.sha256(
            (SCRIPT_DIR / "state8_natural_curriculum_view.py").read_bytes()
        ).hexdigest()
        return namespace

    def load_student_and_teacher(model, adapter_source):
        backbone = original_load(model, adapter_source)
        load_external_teacher(backbone, teacher_path)
        return backbone

    def read(path):
        rows = original_read(path)
        if Path(path).resolve() in {train_path, valid_path}:
            audit = apply_state8_natural_curriculum_view(rows)
            audits.append({"path": str(path), **audit})
        return rows

    def capture(self, *args, **kwargs):
        return capture_from_external_teacher(self, original_capture, *args, **kwargs)

    argparse.ArgumentParser.parse_args = parse
    trainer.load_replicated_data_parallel_adapter = load_student_and_teacher
    trainer.read_jsonl = read
    trainer.dual_path_curriculum_weights = repaired_dual_path_weights
    trainer.causal_lm_loss = weighted_loss
    LayerwiseRecurrentStudentModel.capture_direct_state = capture
    try:
        trainer.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            receipt = {
                "stage": "state8_repaired_teacher_curriculum_v1",
                "teacher_adapter_sha256": teacher_digest,
                "teacher_trainable": False,
                "state_token_count": 8,
                "label_auxiliary_weight": label_weight,
                "trajectory_weight": trajectory_weight,
                "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
                "online_kd_phase_schedule": [0.0, 1.0, 1.0, 1.0, 1.0],
                "endpoint_online_kd_phase_weight": 1.0,
                "final_inference_route": "strict_state_only",
                "dataset_audits": audits,
            }
            (output_dir / "state8_repaired_teacher_curriculum.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.load_replicated_data_parallel_adapter = original_load
        trainer.read_jsonl = original_read
        trainer.dual_path_curriculum_weights = original_dual_weights
        trainer.causal_lm_loss = original_lm_loss
        LayerwiseRecurrentStudentModel.capture_direct_state = original_capture


if __name__ == "__main__":
    main()
