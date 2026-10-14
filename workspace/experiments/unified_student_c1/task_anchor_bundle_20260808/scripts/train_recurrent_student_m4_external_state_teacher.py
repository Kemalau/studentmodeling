#!/usr/bin/env python3
"""Run the frozen recurrent trainer with an independent state-path teacher.

The ordinary trajectory objective captures its direct teacher state from the
same adapter that is being trained.  That is appropriate for recurrent closure
once the adapter already knows the task, but it cannot repair a task-specific
state collapse.  This opt-in entrypoint loads a second, frozen PEFT adapter and
uses it only while capturing the direct 64-layer state trajectory.  Recurrent
updates, state-only decoding, generation loss, and causal losses continue to
use the trainable student adapter.

The feature is deliberately isolated from the hash-frozen production trainer.
It is intended for bounded, explicitly labelled diagnostics until T3 and C4
both pass.  The teacher path and adapter hash are written into the trainer
configuration so an exact-resume checkpoint cannot silently change teachers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from peft.utils import save_and_load as peft_save_and_load

# The production launcher executes this file directly, whereas unit tests import
# it as ``scripts.<module>``.  The frozen trainer and model modules deliberately
# remain runnable scripts with top-level imports, so make their directory
# explicit in both entry modes instead of relying on the caller's PYTHONPATH.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from open_t3_training_view import apply_open_t3_training_view


TEACHER_ADAPTER_NAME = "external_state_teacher"
STUDENT_ADAPTER_NAME = "default"
ENV_PATH = "STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER"
ENV_SHA256 = "STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256"


def external_teacher_spec() -> tuple[Path, str]:
    path_text = os.environ.get(ENV_PATH, "").strip()
    expected = os.environ.get(ENV_SHA256, "").strip().lower()
    if not path_text:
        raise ValueError(f"{ENV_PATH} is required")
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{ENV_SHA256} must be one lowercase SHA256 digest")
    path = Path(path_text)
    weights = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError(f"External state teacher is incomplete: {path}")
    observed = trainer.file_sha256(weights)
    if observed != expected:
        raise ValueError(
            "External state teacher hash drift: "
            f"expected={expected} observed={observed}"
        )
    return path, expected


def cli_path(name: str) -> Path:
    import sys

    if name not in sys.argv:
        raise ValueError(f"Missing required argument {name}")
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} has no value")
    return Path(sys.argv[index + 1]).resolve()


def attach_teacher_config(args: argparse.Namespace, path: Path, digest: str) -> None:
    """Make the external objective explicit in run/checkpoint configuration."""
    if getattr(args, "state_architecture", None) != "layerwise":
        raise ValueError("External state teacher requires layerwise state")
    if float(getattr(args, "direct_replay_weight", -1.0)) != 0.0:
        raise ValueError("External state teacher requires zero direct replay weight")
    if float(getattr(args, "consistency_weight", 0.0)) <= 0.0:
        raise ValueError("External state teacher requires positive consistency weight")
    args.external_state_teacher_adapter = str(path)
    args.external_state_teacher_sha256 = digest
    args.external_state_teacher_adapter_name = TEACHER_ADAPTER_NAME
    args.external_state_teacher_scope = "direct_layerwise_trajectory_only"
    args.external_state_teacher_trainable = False
    args.training_view = "open_t3_no_candidate_enumeration"


def load_external_teacher(backbone: Any, path: Path) -> None:
    if TEACHER_ADAPTER_NAME in backbone.peft_config:
        raise ValueError(f"Adapter name collision: {TEACHER_ADAPTER_NAME}")
    original = peft_save_and_load._maybe_shard_state_dict_for_tp
    peft_save_and_load._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None
    try:
        backbone.load_adapter(
            str(path),
            adapter_name=TEACHER_ADAPTER_NAME,
            is_trainable=False,
        )
    finally:
        peft_save_and_load._maybe_shard_state_dict_for_tp = original
    backbone.set_adapter(STUDENT_ADAPTER_NAME, inference_mode=False)
    teacher_parameters = [
        parameter
        for name, parameter in backbone.named_parameters()
        if f".{TEACHER_ADAPTER_NAME}." in name
    ]
    if not teacher_parameters:
        raise ValueError("External state teacher loaded no named parameters")
    if any(parameter.requires_grad for parameter in teacher_parameters):
        raise ValueError("External state teacher parameters are unexpectedly trainable")
    backbone._studentmodeling_external_state_teacher = TEACHER_ADAPTER_NAME
    backbone._studentmodeling_student_adapter = STUDENT_ADAPTER_NAME


def capture_from_external_teacher(
    wrapper: LayerwiseRecurrentStudentModel,
    original_capture,
    *args,
    **kwargs,
):
    """Capture one no-grad teacher trajectory and restore the student exactly."""
    backbone = wrapper.backbone
    teacher_name = getattr(
        backbone, "_studentmodeling_external_state_teacher", None
    )
    student_name = getattr(backbone, "_studentmodeling_student_adapter", None)
    if teacher_name != TEACHER_ADAPTER_NAME or student_name != STUDENT_ADAPTER_NAME:
        raise ValueError("External state teacher was not attached to the backbone")
    active = backbone.active_adapter
    if active != student_name:
        raise ValueError(f"Expected active student adapter, got {active!r}")
    was_training = backbone.training
    backbone.set_adapter(teacher_name, inference_mode=True)
    backbone.eval()
    try:
        with torch.no_grad():
            return original_capture(wrapper, *args, **kwargs)
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
    training_view_audits: list[dict[str, object]] = []

    def parse_args_with_teacher(parser, *args, **kwargs):
        namespace = original_parse_args(parser, *args, **kwargs)
        attach_teacher_config(namespace, teacher_path, teacher_digest)
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
        return capture_from_external_teacher(
            self, original_capture, *args, **kwargs
        )

    argparse.ArgumentParser.parse_args = parse_args_with_teacher
    trainer.load_replicated_data_parallel_adapter = load_student_and_teacher
    trainer.read_jsonl = read_with_open_t3
    LayerwiseRecurrentStudentModel.capture_direct_state = capture
    try:
        trainer.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if not training_view_audits:
                raise AssertionError("External-teacher run changed no Malrule T3 rows")
            import json

            receipt = {
                "stage": "external_state_teacher_training_view_v1",
                "training_view": "open_t3_no_candidate_enumeration",
                "teacher_adapter": str(teacher_path),
                "teacher_adapter_sha256": teacher_digest,
                "teacher_scope": "direct_layerwise_trajectory_only",
                "teacher_trainable": False,
                **training_view_audits[0],
            }
            (output_dir / "external_state_teacher_training_view.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
        trainer.load_replicated_data_parallel_adapter = original_load
        trainer.read_jsonl = original_read_jsonl
        LayerwiseRecurrentStudentModel.capture_direct_state = original_capture


if __name__ == "__main__":
    main()
