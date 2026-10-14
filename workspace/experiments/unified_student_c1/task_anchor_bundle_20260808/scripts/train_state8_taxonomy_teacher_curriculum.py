#!/usr/bin/env python3
"""Five-stage State8 teacher curriculum with auxiliary taxonomy supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

import train_recurrent_student_m4 as trainer
import train_recurrent_student_m4_state8_external_teacher_curriculum as curriculum
from state8_natural_curriculum_view import apply_state8_natural_curriculum_view
from state8_taxonomy_contract import apply_state8_taxonomy_contract
from train_state8_stateonly_teacher_repair import schema_weighted_loss_factory


def cli_value(name: str) -> str:
    if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
        raise ValueError(f"Missing required argument {name}")
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    output_dir = Path(cli_value("--output-dir")).resolve()
    label_weight = float(os.environ.get("STATE8_CURRICULUM_LABEL_WEIGHT", "4"))
    tokenizer = AutoTokenizer.from_pretrained(
        cli_value("--model"), trust_remote_code=True
    )
    weighted_loss = schema_weighted_loss_factory(tokenizer, label_weight)

    original_parse = argparse.ArgumentParser.parse_args
    original_view = curriculum.apply_state8_natural_curriculum_view
    original_loss = trainer.causal_lm_loss
    audits: list[dict[str, object]] = []

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        namespace.label_auxiliary_weight = label_weight
        namespace.taxonomy_label_contract = "source_aware_no_answer_labels_v1"
        namespace.taxonomy_candidates_visible_to_decoder = False
        namespace.semantic_judge_authoritative = True
        namespace.exact_label_metric_authoritative = False
        return namespace

    def combined_view(rows):
        natural = apply_state8_natural_curriculum_view(rows)
        taxonomy = apply_state8_taxonomy_contract(rows)
        audit = {**natural, **taxonomy}
        audits.append(audit)
        return audit

    argparse.ArgumentParser.parse_args = parse
    curriculum.apply_state8_natural_curriculum_view = combined_view
    trainer.causal_lm_loss = weighted_loss
    try:
        curriculum.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if not audits or sum(
                int(row.get("taxonomy_contract_rows", 0)) for row in audits
            ) <= 0:
                raise RuntimeError("Taxonomy curriculum applied no diagnosis contracts")
            receipt = {
                "stage": "state8_taxonomy_teacher_curriculum_v1",
                "status": "complete",
                "state_token_count": 8,
                "output_schema": "label_diagnosis_evidence_v1",
                "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
                "label_auxiliary_weight": label_weight,
                "taxonomy_contract": "source_aware_no_answer_labels_v1",
                "taxonomy_candidates_visible_to_decoder": False,
                "semantic_judge_authoritative": True,
                "exact_label_metric_authoritative": False,
                "final_inference_route": "strict_state_only",
                "raw_history_visible_to_decoder": False,
                "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audits": audits,
            }
            (output_dir / "taxonomy_teacher_curriculum.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        curriculum.apply_state8_natural_curriculum_view = original_view
        trainer.causal_lm_loss = original_loss


if __name__ == "__main__":
    main()
