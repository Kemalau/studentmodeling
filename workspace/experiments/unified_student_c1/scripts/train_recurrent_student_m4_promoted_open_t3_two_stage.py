#!/usr/bin/env python3
"""Exact-epoch open-T3 trainer with an audited mid-run KD transition."""

from __future__ import annotations

import copy
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
    formation_kd_weight = cli_float("--behavior-kd-weight", 0.0)
    if formation_kd_weight != 0.5:
        raise ValueError("Two-stage open-T3 formation requires behavior KD 0.5")
    if cli_float("--direct-replay-weight", -1.0) != 0.0:
        raise ValueError("Two-stage open-T3 training requires zero direct replay")

    os.environ["STUDENT_TRAINING_VIEW"] = "open_t3_selective_kd"
    train_path = str(Path(cli_value("--train-jsonl")).resolve())
    output_dir = Path(cli_value("--output-dir"))
    original_read_jsonl = trainer.read_jsonl
    original_objective = trainer.backward_training_objective
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

    def objective_with_dynamic_selective_kd(
        wrapper,
        tokenizer,
        row,
        causal_row,
        cache,
        args,
        device,
        state_token_id,
        loss_scale,
        training_progress,
    ):
        global_weight = float(args.behavior_kd_weight)
        effective_weight = (
            0.0
            if row.get("task") == "diagnose_misconception"
            and row.get("source") == "malrule"
            else global_weight
        )
        local_args = copy.copy(args)
        local_args.behavior_kd_weight = effective_weight
        record = original_objective(
            wrapper,
            tokenizer,
            row,
            causal_row,
            cache,
            local_args,
            device,
            state_token_id,
            loss_scale,
            training_progress,
        )
        record["behavior_kd_weight_global"] = global_weight
        record["behavior_kd_weight_effective"] = effective_weight
        record["open_t3_selective_kd_excluded"] = (
            global_weight > 0.0 and effective_weight == 0.0
        )
        return record

    trainer.read_jsonl = read_with_open_t3
    trainer.backward_training_objective = objective_with_dynamic_selective_kd
    trainer.main()
    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
        return
    if not audits:
        raise AssertionError("Two-stage open-T3 view changed no rows")
    history = json.loads(
        (output_dir / "trainer_history.json").read_text(encoding="utf-8")
    )
    excluded = sum(bool(row.get("open_t3_selective_kd_excluded")) for row in history)
    retained = sum(
        float(row.get("behavior_kd_weight_effective", -1.0))
        == formation_kd_weight
        for row in history
    )
    zero_phase = sum(
        float(row.get("behavior_kd_weight_global", -1.0)) == 0.0
        for row in history
    )
    if excluded <= 0 or retained <= 0:
        raise AssertionError(
            f"Two-stage selective KD audit failed: excluded={excluded} retained={retained}"
        )
    receipt = {
        "stage": "promoted_open_t3_selective_kd_training_view_v1",
        "scope": "unseen_exact_epoch_suffix_only",
        "source_train_jsonl": train_path,
        "global_behavior_kd_weight": formation_kd_weight,
        "excluded_behavior_kd_weight": 0.0,
        "direct_replay_weight": 0.0,
        "excluded_history_rows": excluded,
        "retained_history_rows": retained,
        "zero_kd_consolidation_history_rows": zero_phase,
        "dynamic_two_stage_kd": True,
        "causal_rows_modified": False,
        "validation_rows_modified": False,
        **audits[0],
    }
    (output_dir / "promoted_open_t3_selective_kd_training_view.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
