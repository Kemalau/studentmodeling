#!/usr/bin/env python3
"""State-only repair with full-history teacher KD and optional trajectory alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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


def cli_value(name: str) -> str:
    if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
        raise ValueError(f"Missing required argument {name}")
    return sys.argv[sys.argv.index(name) + 1]


def find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    for index in range(start, len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


def schema_weighted_loss_factory(tokenizer, label_weight: float):
    if label_weight < 1.0:
        raise ValueError("Label weight must be at least one")
    opener = tokenizer("<label>", add_special_tokens=False)["input_ids"]
    closer = tokenizer("</label>", add_special_tokens=False)["input_ids"]
    if not opener or not closer:
        raise ValueError("Tokenizer produced an empty label delimiter")

    def schema_weighted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shifted_labels = labels[:, 1:].contiguous()
        supervised = shifted_labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("Causal LM batch has no supervised target tokens")
        selected_logits = logits[:, :-1][supervised].float()
        selected_labels = shifted_labels[supervised]
        token_losses = F.cross_entropy(
            selected_logits, selected_labels, reduction="none"
        )
        weights = token_losses.new_ones(token_losses.shape)
        offset = 0
        for row in shifted_labels:
            values = row[row.ne(-100)].tolist()
            begin = find_subsequence(values, opener)
            end = find_subsequence(values, closer, begin + len(opener)) if begin >= 0 else -1
            if begin >= 0 and end >= 0:
                # Weight the canonical label value, not the XML delimiters.
                start_value = offset + begin + len(opener)
                end_value = offset + end
                weights[start_value:end_value] = float(label_weight)
            offset += len(values)
        return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)

    schema_weighted_loss.label_opener_ids = opener
    schema_weighted_loss.label_closer_ids = closer
    schema_weighted_loss.label_weight = float(label_weight)
    return schema_weighted_loss


def main() -> None:
    teacher_path, teacher_digest = external_teacher_spec()
    train_path = cli_path("--train-jsonl")
    valid_path = cli_path("--valid-jsonl")
    output_dir = cli_path("--output-dir")
    trajectory_weight = float(os.environ.get("STATE8_REPAIR_TRAJECTORY_WEIGHT", "0"))
    label_weight = float(os.environ.get("STATE8_REPAIR_LABEL_WEIGHT", "4"))
    if trajectory_weight < 0:
        raise ValueError("Trajectory weight must be non-negative")
    tokenizer = AutoTokenizer.from_pretrained(
        cli_value("--model"), trust_remote_code=True
    )
    weighted_loss = schema_weighted_loss_factory(tokenizer, label_weight)

    original_parse = argparse.ArgumentParser.parse_args
    original_load = trainer.load_replicated_data_parallel_adapter
    original_read = trainer.read_jsonl
    original_capture = LayerwiseRecurrentStudentModel.capture_direct_state
    original_history_dropout = trainer.history_dropout_at_progress
    original_dual_weights = trainer.dual_path_curriculum_weights
    original_lm_loss = trainer.causal_lm_loss
    audits: list[dict[str, int | str]] = []

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        if namespace.state_architecture != "layerwise" or namespace.state_token_count != 8:
            raise ValueError("State-only repair requires eight-slot layerwise state")
        if not namespace.history_dropout_curriculum or not namespace.dual_path_curriculum:
            raise ValueError("State-only repair requires explicit curriculum flags")
        if namespace.online_kd_weight <= 0:
            raise ValueError("State-only repair requires positive teacher KD")
        if namespace.behavior_kd_weight != 0.0 or namespace.direct_replay_weight != 0.0:
            raise ValueError("State-only repair forbids cached KD and direct replay")
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
        namespace.curriculum_route = "stateonly_consolidation_after_teacher_curriculum"
        namespace.history_dropout_schedule = [1.0]
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
    trainer.history_dropout_at_progress = lambda _args, _progress: 1.0
    trainer.dual_path_curriculum_weights = lambda _args, _progress: (0.0, 1.0, 1.0)
    trainer.causal_lm_loss = weighted_loss
    LayerwiseRecurrentStudentModel.capture_direct_state = capture
    try:
        trainer.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            receipt = {
                "stage": "state8_stateonly_teacher_repair_v1",
                "teacher_adapter_sha256": teacher_digest,
                "teacher_trainable": False,
                "state_token_count": 8,
                "state_ce_weight": 1.0,
                "online_kd_phase_weight": 1.0,
                "trajectory_weight": trajectory_weight,
                "label_auxiliary_weight": label_weight,
                "history_dropout_schedule": [1.0],
                "final_inference_route": "strict_state_only",
                "dataset_audits": audits,
            }
            (output_dir / "state8_stateonly_teacher_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.load_replicated_data_parallel_adapter = original_load
        trainer.read_jsonl = original_read
        trainer.history_dropout_at_progress = original_history_dropout
        trainer.dual_path_curriculum_weights = original_dual_weights
        trainer.causal_lm_loss = original_lm_loss
        LayerwiseRecurrentStudentModel.capture_direct_state = original_capture


if __name__ == "__main__":
    main()
