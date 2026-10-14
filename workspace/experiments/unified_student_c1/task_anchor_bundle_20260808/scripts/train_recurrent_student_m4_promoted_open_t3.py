#!/usr/bin/env python3
"""Run the promoted exact-epoch trainer with the audited open Malrule T3 view."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import train_recurrent_student_m4_promoted as trainer
from open_t3_training_view import apply_open_t3_training_view


def cli_float(name: str, default: float) -> float:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} has no value")
    return float(sys.argv[index + 1])


def cli_value(name: str) -> str:
    if name not in sys.argv:
        raise ValueError(f"Missing required argument {name}")
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} has no value")
    return sys.argv[index + 1]


def main() -> None:
    if cli_float("--behavior-kd-weight", 0.0) != 0.0:
        raise ValueError("Promoted open T3 requires zero behavior KD")
    if cli_float("--direct-replay-weight", 0.0) != 0.0:
        raise ValueError("Promoted open T3 requires zero direct replay")

    os.environ["STUDENT_TRAINING_VIEW"] = "open_t3_no_candidate_enumeration"
    train_path = str(Path(cli_value("--train-jsonl")).resolve())
    output_dir = Path(cli_value("--output-dir"))
    original_read_jsonl = trainer.read_jsonl
    audits: list[dict[str, object]] = []

    def read_with_open_t3(path):
        rows = original_read_jsonl(path)
        if str(Path(path).resolve()) != train_path:
            return rows
        audit = apply_open_t3_training_view(rows)
        if audit["changed_malrule_t3_rows"]:
            audits.append({"path": str(path), **audit})
            print(json.dumps({"open_t3_training_view": audits[-1]}, sort_keys=True), flush=True)
        return rows

    trainer.read_jsonl = read_with_open_t3
    trainer.main()
    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
        return
    if not audits:
        raise AssertionError("Promoted open T3 view changed no rows")
    receipt = {
        "stage": "promoted_open_t3_training_view_v1",
        "scope": "unseen_exact_epoch_suffix_only",
        "source_train_jsonl": train_path,
        "behavior_kd_weight": 0.0,
        "direct_replay_weight": 0.0,
        "causal_rows_modified": False,
        "validation_rows_modified": False,
        **audits[0],
    }
    (output_dir / "promoted_open_t3_training_view.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
