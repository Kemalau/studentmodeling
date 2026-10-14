#!/usr/bin/env python3
"""Exact final fit with matched Teacher provenance and same-question routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import train_recurrent_student_m4 as trainer
import train_state8_diagnosis_first_source_anchored_repair as source_anchor


PROVENANCE_STAGE = "state8_diagnosis_first_teacher_full_provenance_gate_v1"
OUTPUT_SCHEMA = "diagnosis_evidence_label_v1"


def required_sha(name: str) -> str:
    value = os.environ.get(name, "").strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be one lowercase SHA256 digest")
    return value


def main() -> None:
    output_dir: Path | None = None
    teacher_eval_sha = required_sha("STATE8_MATCHED_FULL_TEACHER_EVAL_SHA256")
    teacher_gate_sha = required_sha("STATE8_MATCHED_FULL_TEACHER_GATE_SHA256")
    teacher_provenance_sha = required_sha(
        "STATE8_MATCHED_FULL_TEACHER_PROVENANCE_SHA256"
    )
    original_parse = argparse.ArgumentParser.parse_args
    original_trainer_main = trainer.main

    def parse_outer(parser, *args, **kwargs):
        nonlocal output_dir
        namespace = original_parse(parser, *args, **kwargs)
        output_dir = Path(namespace.output_dir)
        return namespace

    def matched_trainer_main() -> None:
        nested_parse = argparse.ArgumentParser.parse_args

        def parse_matched(parser, *args, **kwargs):
            namespace = nested_parse(parser, *args, **kwargs)
            namespace.teacher_prompt_provenance_stage = PROVENANCE_STAGE
            namespace.teacher_evaluation_input_sha256 = teacher_eval_sha
            namespace.teacher_clean_prompt_sha256 = teacher_eval_sha
            namespace.teacher_gate_sha256 = teacher_gate_sha
            namespace.teacher_provenance_gate_sha256 = teacher_provenance_sha
            namespace.teacher_output_schema = OUTPUT_SCHEMA
            namespace.output_schema = OUTPUT_SCHEMA
            namespace.teacher_diagnosis_field_order = ["diagnosis", "evidence", "label"]
            namespace.diagnosis_field_order = ["diagnosis", "evidence", "label"]
            namespace.canonical_label_role = "auxiliary_ordinary_weight"
            namespace.teacher_gate_support = 1494
            namespace.legacy_answer_adjacent_contracts = 0
            namespace.matched_teacher_required = True
            namespace.full_teacher_gate_required = True
            namespace.same_question_finalfit = True
            namespace.entrypoint_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            return namespace

        argparse.ArgumentParser.parse_args = parse_matched
        try:
            original_trainer_main()
        finally:
            argparse.ArgumentParser.parse_args = nested_parse

    argparse.ArgumentParser.parse_args = parse_outer
    trainer.main = matched_trainer_main
    try:
        source_anchor.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if output_dir is None:
                raise RuntimeError("Same-question final fit did not observe output dir")
            config = json.loads((output_dir / "run_config.json").read_text())
            source_receipt = json.loads(
                (output_dir / "diagnosis_first_source_anchored_repair.json").read_text()
            )
            if config.get("teacher_prompt_provenance_stage") != PROVENANCE_STAGE:
                raise RuntimeError("Matched full Teacher provenance was not serialized")
            if config.get("output_schema") != OUTPUT_SCHEMA:
                raise RuntimeError("Diagnosis-first output schema was not serialized")
            if source_receipt.get("current_query_routing") != (
                "current_item_visible_response_state_only_v1:eedi_only"
            ):
                raise RuntimeError("Eedi current-query routing was not serialized")
            if source_receipt.get("student_response_visible_to_decoder") is not False:
                raise RuntimeError("Student response leaked to decoder")
            receipt = {
                "stage": "state8_eedi_same_question_finalfit_v1",
                "status": "complete",
                "teacher_prompt_provenance_stage": PROVENANCE_STAGE,
                "teacher_evaluation_input_sha256": teacher_eval_sha,
                "teacher_gate_sha256": teacher_gate_sha,
                "teacher_provenance_gate_sha256": teacher_provenance_sha,
                "teacher_output_schema": OUTPUT_SCHEMA,
                "teacher_gate_support": 1494,
                "state_token_count": 8,
                "history_dropout_schedule": [1.0],
                "raw_history_visible_to_decoder": False,
                "final_inference_route": "strict_state_only",
                "current_query_routing": source_receipt["current_query_routing"],
                "current_item_visible_to_decoder": True,
                "student_response_visible_to_decoder": False,
                "diagnosis_field_order": ["diagnosis", "evidence", "label"],
                "canonical_label_role": "auxiliary_ordinary_weight",
                "eedi_anchor_schedule": source_receipt["eedi_anchor_schedule"],
                "malrule_anchor_schedule": source_receipt["malrule_anchor_schedule"],
                "persistence_anchor_schedule": source_receipt[
                    "persistence_anchor_schedule"
                ],
                "trajectory_cosine_weight": source_receipt[
                    "trajectory_cosine_weight"
                ],
                "trajectory_relative_mse_weight": source_receipt[
                    "trajectory_relative_mse_weight"
                ],
                "causal_teacher_trajectory_weight": float(
                    config.get("causal_teacher_trajectory_weight", 0.0)
                ),
                "early_label_binding_weight": float(
                    config.get("early_label_binding_weight", 0.0)
                ),
                "binding_constraints_inherited_from_promotion": True,
                "same_question_causal_replay_required": True,
                "semantic_judge_authoritative": True,
                "exact_match_authoritative": False,
                "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            (output_dir / "eedi_same_question_finalfit.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.main = original_trainer_main


if __name__ == "__main__":
    main()
