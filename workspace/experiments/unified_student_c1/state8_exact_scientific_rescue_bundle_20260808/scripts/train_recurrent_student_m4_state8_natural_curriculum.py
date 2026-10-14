#!/usr/bin/env python3
"""Eight-slot natural-language T3 with full-history to state-only curriculum."""

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
from state8_natural_curriculum_view import apply_state8_natural_curriculum_view


def cli_path(name: str) -> Path:
    if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
        raise ValueError(f"Missing required argument {name}")
    return Path(sys.argv[sys.argv.index(name) + 1]).resolve()


def main() -> None:
    train_path = cli_path("--train-jsonl")
    valid_path = cli_path("--valid-jsonl")
    output_dir = cli_path("--output-dir")
    original_parse = argparse.ArgumentParser.parse_args
    original_read = trainer.read_jsonl
    audits: list[dict[str, int | str]] = []

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        if namespace.state_architecture != "layerwise":
            raise ValueError("Eight-slot curriculum requires layerwise state")
        if namespace.state_token_count != 8:
            raise ValueError("Eight-slot curriculum requires --state-token-count 8")
        if not namespace.history_dropout_curriculum:
            raise ValueError("Eight-slot curriculum requires history dropout")
        if not namespace.dual_path_curriculum:
            raise ValueError("Eight-slot curriculum requires dual teacher/student paths")
        if namespace.online_kd_weight <= 0:
            raise ValueError("Eight-slot curriculum requires positive online KD")
        if namespace.behavior_kd_weight != 0.0:
            raise ValueError("Rewritten natural targets are incompatible with cached KD")
        namespace.output_schema = "label_diagnosis_evidence_v1"
        namespace.history_dropout_schedule = [0.0, 0.25, 0.5, 0.75, 1.0]
        namespace.curriculum_route = "dual_path_weighted_teacher_to_student"
        namespace.online_teacher_scope = "same_sample_full_history_answer_logits"
        namespace.final_inference_route = "strict_state_only"
        namespace.state8_natural_view_sha256 = hashlib.sha256(
            (SCRIPT_DIR / "state8_natural_curriculum_view.py").read_bytes()
        ).hexdigest()
        namespace.state8_curriculum_entrypoint_sha256 = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        return namespace

    def read(path):
        rows = original_read(path)
        if Path(path).resolve() in {train_path, valid_path}:
            audit = apply_state8_natural_curriculum_view(rows)
            audits.append({"path": str(path), **audit})
        return rows

    argparse.ArgumentParser.parse_args = parse
    trainer.read_jsonl = read
    try:
        trainer.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            by_path = {str(row["path"]): row for row in audits}
            for expected in (train_path, valid_path):
                audit = by_path.get(str(expected))
                if not audit or int(audit["changed_t3_rows"]) <= 0:
                    raise AssertionError(
                        f"Natural curriculum changed no T3 rows in {expected}"
                    )
            receipt = {
                "stage": "state8_natural_history_dropout_curriculum_v1",
                "state_token_count": 8,
                "output_schema": "label_diagnosis_evidence_v1",
                "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
                "curriculum_route": "dual_path_weighted_teacher_to_student",
                "online_kd_enabled": True,
                "final_inference_route": "strict_state_only",
                "dataset_audits": audits,
            }
            (output_dir / "state8_natural_curriculum_view.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.read_jsonl = original_read


if __name__ == "__main__":
    main()
