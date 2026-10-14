#!/usr/bin/env python3
"""Open Malrule T3 training with KD retained on every compatible task row.

The frozen teacher cache was produced with candidate-enumerating Malrule T3
prompts.  Reusing those logits after removing the candidates is invalid, but
turning KD off globally also removes useful retention supervision from T1,
T2, Eedi T3, T5, and T6.  This audited wrapper changes the prompt and effective
KD weight only for Malrule T3 rows; the frozen trainer remains unmodified.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import train_recurrent_student_m4 as trainer
from open_t3_training_view import (
    apply_open_t3_training_view,
    selective_behavior_kd_weight,
)


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
    global_kd_weight = cli_float("--behavior-kd-weight", 0.0)
    if global_kd_weight <= 0.0:
        raise ValueError("Selective open-T3 KD requires positive behavior KD")
    if cli_float("--direct-replay-weight", 0.0) != 0.0:
        raise ValueError("Selective open-T3 KD requires zero direct replay")

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

    def objective_with_selective_kd(
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
        effective_weight = selective_behavior_kd_weight(
            row, float(args.behavior_kd_weight)
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
        record["behavior_kd_weight_effective"] = effective_weight
        record["open_t3_selective_kd_excluded"] = effective_weight == 0.0
        return record

    trainer.read_jsonl = read_with_open_t3
    trainer.backward_training_objective = objective_with_selective_kd
    trainer.main()
    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
        return
    if not audits:
        raise AssertionError("Selective open-T3 training view changed no rows")
    history = json.loads((output_dir / "trainer_history.json").read_text(encoding="utf-8"))
    excluded = sum(bool(row.get("open_t3_selective_kd_excluded")) for row in history)
    retained = sum(
        float(row.get("behavior_kd_weight_effective", -1.0)) == global_kd_weight
        for row in history
    )
    if excluded <= 0 or retained <= 0 or excluded + retained != len(history):
        raise AssertionError(
            f"Selective KD audit failed: excluded={excluded} retained={retained} "
            f"history={len(history)}"
        )
    receipt = {
        "stage": "open_t3_selective_kd_training_view_v1",
        "scope": "train_jsonl_only",
        "source_train_jsonl": train_path,
        "global_behavior_kd_weight": global_kd_weight,
        "excluded_behavior_kd_weight": 0.0,
        "direct_replay_weight": 0.0,
        "excluded_history_rows": excluded,
        "retained_history_rows": retained,
        "causal_rows_modified": False,
        "validation_rows_modified": False,
        **audits[0],
    }
    (output_dir / "open_t3_selective_kd_training_view.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
