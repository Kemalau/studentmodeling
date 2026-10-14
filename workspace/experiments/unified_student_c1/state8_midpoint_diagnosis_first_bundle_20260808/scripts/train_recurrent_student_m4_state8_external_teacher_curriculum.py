#!/usr/bin/env python3
"""Eight-slot curriculum distilled from a separately qualified T3 teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from state8_natural_curriculum_view import apply_state8_natural_curriculum_view
from train_recurrent_student_m4_external_state_teacher import (
    STUDENT_ADAPTER_NAME,
    TEACHER_ADAPTER_NAME,
    capture_from_external_teacher,
    cli_path,
    external_teacher_spec,
    load_external_teacher,
)


def ensure_dataset_audits(
    audits: list[dict[str, int | str]],
    expected_paths: tuple[Path, ...],
    *,
    read_rows,
    apply_view,
) -> None:
    """Reconstruct provenance when a complete-checkpoint finalizer reads no data.

    Exact distributed recovery can enter the base trainer at the final round.
    In that case no training/validation rows need to be consumed again, so the
    normal ``trainer.read_jsonl`` hook may never populate ``audits``.  The
    model is already complete, but treating the empty process-local hook as a
    failed scientific audit makes every finalization attempt fail after all
    optimizer work has finished.

    Re-reading the same immutable inputs here changes no weights or optimizer
    state.  It only replays the deterministic natural-view transform so the
    receipt records the same dataset-level provenance as a fresh run.
    """
    by_path = {str(row["path"]): row for row in audits}
    for expected in expected_paths:
        key = str(expected)
        if int((by_path.get(key) or {}).get("changed_t3_rows", 0)) > 0:
            continue
        rows = read_rows(expected)
        audit = {
            "path": key,
            "audit_provenance": "reconstructed_from_immutable_input_after_exact_resume",
            **apply_view(rows),
        }
        audits.append(audit)
        by_path[key] = audit


def main() -> None:
    teacher_path, teacher_digest = external_teacher_spec()
    train_path = cli_path("--train-jsonl")
    valid_path = cli_path("--valid-jsonl")
    output_dir = cli_path("--output-dir")
    original_parse = argparse.ArgumentParser.parse_args
    original_load = trainer.load_replicated_data_parallel_adapter
    original_read = trainer.read_jsonl
    original_capture = LayerwiseRecurrentStudentModel.capture_direct_state
    use_cached_online_kd = "--online-kd-cache" in sys.argv
    audits: list[dict[str, int | str]] = []

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        if namespace.state_token_count != 8:
            raise ValueError("External-teacher curriculum requires eight state tokens")
        if not namespace.history_dropout_curriculum or not namespace.dual_path_curriculum:
            raise ValueError("External-teacher curriculum requires dual-path history dropout")
        if namespace.online_kd_weight <= 0:
            raise ValueError("External-teacher curriculum requires positive answer-logit KD")
        if namespace.behavior_kd_weight != 0.0 or namespace.direct_replay_weight != 0.0:
            raise ValueError("External-teacher curriculum forbids cached KD and direct replay")
        if namespace.consistency_weight != 0.0:
            raise ValueError(
                "Qualified-teacher curriculum distills verified answer logits, not an unverified carrier trajectory"
            )
        namespace.external_state_teacher_adapter = str(teacher_path)
        namespace.external_state_teacher_sha256 = teacher_digest
        namespace.external_state_teacher_adapter_name = TEACHER_ADAPTER_NAME
        namespace.external_state_teacher_scope = "qualified_full_history_answer_logits_only"
        namespace.external_state_teacher_trainable = False
        namespace.training_view = "state8_natural_label_diagnosis_evidence"
        namespace.output_schema = "label_diagnosis_evidence_v1"
        namespace.history_dropout_schedule = [0.0, 0.25, 0.5, 0.75, 1.0]
        namespace.curriculum_route = "qualified_frozen_teacher_to_state8_student"
        namespace.online_teacher_scope = (
            "qualified_external_full_history_answer_logits_cache"
            if use_cached_online_kd
            else "qualified_external_full_history_answer_logits"
        )
        namespace.trajectory_teacher_scope = "none_unverified_carrier_trajectory_excluded"
        namespace.teacher_gate_required = True
        namespace.teacher_trainable = False
        namespace.final_inference_route = "strict_state_only"
        # The base trainer derives this field immediately before the first
        # distributed round.  Seed the ordinary (non-diagnostic) value here so
        # an exact-resume preflight compares ``None`` with ``None`` instead of
        # treating an absent derived field as scientific recipe drift.
        namespace.diagnostic_segment_rounds = None
        namespace.state8_natural_view_sha256 = hashlib.sha256(
            (SCRIPT_DIR / "state8_natural_curriculum_view.py").read_bytes()
        ).hexdigest()
        namespace.teacher_prompt_provenance_stage = (
            "state8_clean_prompt_teacher_provenance_gate_v2"
        )
        namespace.teacher_evaluation_input_sha256 = (
            "78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5"
        )
        namespace.teacher_clean_prompt_sha256 = (
            "c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314"
        )
        namespace.legacy_answer_adjacent_contracts = 0
        namespace.entrypoint_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return namespace

    def load_student_and_teacher(model, adapter_source):
        backbone = original_load(model, adapter_source)
        if not use_cached_online_kd:
            load_external_teacher(backbone, teacher_path)
        return backbone

    def read(path):
        rows = original_read(path)
        resolved_path = Path(path).resolve()
        if resolved_path in {train_path, valid_path}:
            audit = apply_state8_natural_curriculum_view(rows)
            # ``cli_path`` resolves symlinks, while the trainer deliberately
            # keeps the logical command-line path when reading immutable
            # cached data.  Store the same canonical identity on both sides;
            # otherwise the post-training receipt can reject a successfully
            # completed exact epoch merely because its input was a symlink.
            audits.append({"path": str(resolved_path), **audit})
        return rows

    def capture(self, *args, **kwargs):
        if use_cached_online_kd:
            return original_capture(self, *args, **kwargs)
        return capture_from_external_teacher(self, original_capture, *args, **kwargs)

    argparse.ArgumentParser.parse_args = parse
    trainer.load_replicated_data_parallel_adapter = load_student_and_teacher
    trainer.read_jsonl = read
    LayerwiseRecurrentStudentModel.capture_direct_state = capture
    try:
        trainer.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            ensure_dataset_audits(
                audits,
                (train_path, valid_path),
                read_rows=original_read,
                apply_view=apply_state8_natural_curriculum_view,
            )
            by_path = {str(row["path"]): row for row in audits}
            for expected in (train_path, valid_path):
                if int((by_path.get(str(expected)) or {}).get("changed_t3_rows", 0)) <= 0:
                    raise AssertionError(f"No natural T3 rows in {expected}")
            receipt = {
                "stage": "state8_qualified_external_teacher_curriculum_v1",
                "teacher_adapter": str(teacher_path),
                "teacher_adapter_sha256": teacher_digest,
                "teacher_gate_required": True,
                "teacher_trainable": False,
                "state_token_count": 8,
                "output_schema": "label_diagnosis_evidence_v1",
                "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
                "online_teacher_scope": (
                    "qualified_external_full_history_answer_logits_cache"
                    if use_cached_online_kd
                    else "qualified_external_full_history_answer_logits"
                ),
                "trajectory_teacher_scope": "none_unverified_carrier_trajectory_excluded",
                "final_inference_route": "strict_state_only",
                "state8_natural_view_sha256": hashlib.sha256(
                    (SCRIPT_DIR / "state8_natural_curriculum_view.py").read_bytes()
                ).hexdigest(),
                "teacher_prompt_provenance_stage": "state8_clean_prompt_teacher_provenance_gate_v2",
                "teacher_evaluation_input_sha256": "78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5",
                "teacher_clean_prompt_sha256": "c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314",
                "legacy_answer_adjacent_contracts": 0,
                "dataset_audits": audits,
            }
            (output_dir / "state8_external_teacher_curriculum.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.load_replicated_data_parallel_adapter = original_load
        trainer.read_jsonl = original_read
        LayerwiseRecurrentStudentModel.capture_direct_state = original_capture


if __name__ == "__main__":
    main()
