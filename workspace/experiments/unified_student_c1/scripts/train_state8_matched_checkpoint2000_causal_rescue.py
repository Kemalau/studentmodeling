#!/usr/bin/env python3
"""Schema-matched strict-state-only rescue from checkpoint-2000."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import train_recurrent_student_m4 as trainer
import train_state8_diagnosis_first_causal_repair as diagnosis
import train_state8_diagnosis_first_task_anchored_causal_repair as anchored


OUTPUT_SCHEMA = "diagnosis_evidence_label_v1"
PROVENANCE_STAGE = "state8_diagnosis_first_teacher_full_provenance_gate_v1"


def main() -> None:
    mode = os.environ.get("STATE8_RESCUE_MODE", "").strip()
    if mode not in {"semantic_preserve", "causal_taskanchor"}:
        raise ValueError(f"Unsupported STATE8_RESCUE_MODE={mode!r}")
    selected = diagnosis if mode == "semantic_preserve" else anchored
    output_dir: Path | None = None
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
            namespace.output_schema = OUTPUT_SCHEMA
            namespace.teacher_output_schema = OUTPUT_SCHEMA
            namespace.diagnosis_field_order = ["diagnosis", "evidence", "label"]
            namespace.teacher_diagnosis_field_order = ["diagnosis", "evidence", "label"]
            namespace.canonical_label_role = "auxiliary_ordinary_weight"
            namespace.semantic_judge_authoritative = True
            namespace.exact_match_authoritative = False
            namespace.teacher_prompt_provenance_stage = PROVENANCE_STAGE
            namespace.final_inference_route = "strict_state_only"
            namespace.raw_history_visible_to_decoder = False
            namespace.matched_checkpoint2000_rescue_mode = mode
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
        selected.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if output_dir is None:
                raise RuntimeError("Matched rescue did not observe output directory")
            config = json.loads((output_dir / "run_config.json").read_text())
            if config.get("output_schema") != OUTPUT_SCHEMA:
                raise RuntimeError("Matched diagnosis-first output schema drift")
            if config.get("raw_history_visible_to_decoder") is not False:
                raise RuntimeError("Matched rescue exposed raw history")
            if config.get("final_inference_route") != "strict_state_only":
                raise RuntimeError("Matched rescue inference route drift")
            if not (output_dir / "diagnosis_first_causal_repair.json").is_file():
                raise RuntimeError("Diagnosis-first repair receipt missing")
            if mode == "causal_taskanchor" and not (
                output_dir / "diagnosis_first_task_anchored_causal_repair.json"
            ).is_file():
                raise RuntimeError("Task-anchor receipt missing")
            receipt = {
                "stage": "state8_matched_checkpoint2000_causal_rescue_training_v1",
                "status": "complete",
                "rescue_mode": mode,
                "source_checkpoint_step": 2000,
                "state_token_count": 8,
                "history_dropout_schedule": [1.0],
                "output_schema": OUTPUT_SCHEMA,
                "diagnosis_field_order": ["diagnosis", "evidence", "label"],
                "canonical_label_role": "auxiliary_ordinary_weight",
                "semantic_judge_authoritative": True,
                "exact_match_authoritative": False,
                "teacher_prompt_provenance_stage": PROVENANCE_STAGE,
                "raw_history_visible_to_decoder": False,
                "final_inference_route": "strict_state_only",
                "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            (output_dir / "matched_checkpoint2000_causal_rescue.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.main = original_trainer_main


if __name__ == "__main__":
    main()
