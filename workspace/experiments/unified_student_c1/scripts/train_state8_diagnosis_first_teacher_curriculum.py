#!/usr/bin/env python3
"""Five-stage State8 teacher curriculum with diagnosis-first supervision.

This combines the qualified full-history teacher curriculum with the
source-aware taxonomy contract, while making natural-language diagnosis and
evidence the primary decoder target.  The canonical label is retained as an
ordinary-weight auxiliary field at the end of the target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import train_state8_taxonomy_teacher_curriculum as curriculum
import train_recurrent_student_m4 as trainer
from train_state8_diagnosis_first_causal_repair import (
    FIELD_ORDER,
    apply_diagnosis_first_order,
)


def main() -> None:
    output_dir: Path | None = None
    audits: list[dict[str, Any]] = []
    original_parse = argparse.ArgumentParser.parse_args
    original_view = curriculum.apply_state8_natural_curriculum_view
    original_gather = trainer.gather_distributed_history

    def parse(parser, *args, **kwargs):
        nonlocal output_dir
        namespace = original_parse(parser, *args, **kwargs)
        output_dir = Path(namespace.output_dir)
        namespace.diagnosis_field_order = list(FIELD_ORDER)
        namespace.semantic_fields_primary = True
        namespace.canonical_label_role = "auxiliary_ordinary_weight"
        namespace.label_auxiliary_weight = 1.0
        namespace.final_inference_route = "strict_state_only"
        return namespace

    def diagnosis_first_view(rows):
        audit = dict(original_view(rows))
        order = apply_diagnosis_first_order(rows)
        audit.update(order)
        audits.append(audit)
        return audit

    def gather_without_single_resume_duplication(local_history, history_base, context):
        """Remove the second copy of an exact single-device resume prefix.

        The base trainer keeps the checkpoint prefix in ``local_history`` and
        also passes an identical copy as ``history_base``.  Distributed
        recovery needs the latter because rank-local histories restart empty;
        single-device recovery does not.  Restrict this repair to the exact
        prefix identity so any different history shape still fails closed.
        """
        if not context.enabled and history_base:
            prefix = local_history[: len(history_base)]
            if prefix != history_base:
                raise RuntimeError(
                    "Single-device diagnosis-first resume history prefix drift"
                )
            return original_gather(local_history, [], context)
        return original_gather(local_history, history_base, context)

    argparse.ArgumentParser.parse_args = parse
    curriculum.apply_state8_natural_curriculum_view = diagnosis_first_view
    trainer.gather_distributed_history = gather_without_single_resume_duplication
    try:
        curriculum.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if output_dir is None:
                raise RuntimeError("Diagnosis-first curriculum did not observe output directory")
            target_rows = sum(
                int(item.get("diagnosis_first_target_rows", 0)) for item in audits
            )
            prompt_rows = sum(
                int(item.get("diagnosis_first_prompt_rows", 0)) for item in audits
            )
            if target_rows <= 0 or prompt_rows <= 0:
                raise RuntimeError(
                    "Diagnosis-first curriculum changed no supervised targets or prompts"
                )
            receipt = {
                "stage": "state8_diagnosis_first_teacher_curriculum_v1",
                "status": "complete",
                "state_token_count": 8,
                "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
                "field_order": list(FIELD_ORDER),
                "semantic_fields_primary": True,
                "canonical_label_role": "auxiliary_ordinary_weight",
                "label_auxiliary_weight": 1.0,
                "semantic_judge_authoritative": True,
                "exact_label_metric_authoritative": False,
                "raw_history_visible_to_decoder": False,
                "final_inference_route": "strict_state_only",
                "single_device_resume_history_deduplication": "exact_prefix_only_v1",
                "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audits": audits,
            }
            (output_dir / "diagnosis_first_teacher_curriculum.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        curriculum.apply_state8_natural_curriculum_view = original_view
        trainer.gather_distributed_history = original_gather


if __name__ == "__main__":
    main()
