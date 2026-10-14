#!/usr/bin/env python3
"""Exact-pipeline wrapper for single-device diagnosis-repair resume history.

The shared diagnosis-repair entrypoint remains byte-for-byte immutable for
older experiments.  This wrapper fixes only the single-device completion
audit: the base trainer retains a checkpoint prefix in ``local_history`` and
also supplies the identical prefix as ``history_base``.  We remove the second
copy only after exact row-wise equality has been proven.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_diagnosis_first_causal_repair as diagnosis


def gather_without_single_resume_duplication(
    original_gather: Callable[..., list[dict[str, Any]] | None],
    local_history: list[dict[str, Any]],
    history_base: list[dict[str, Any]],
    context,
) -> list[dict[str, Any]] | None:
    if not context.enabled and history_base:
        prefix = local_history[: len(history_base)]
        if prefix != history_base:
            raise RuntimeError(
                "Single-device diagnosis causal-repair resume history prefix drift"
            )
        return original_gather(local_history, [], context)
    return original_gather(local_history, history_base, context)


def cli_path(name: str) -> Path:
    if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
        raise ValueError(f"Missing required argument {name}")
    return Path(sys.argv[sys.argv.index(name) + 1])


def main() -> None:
    output_dir = cli_path("--output-dir")
    original_gather = trainer.gather_distributed_history
    trainer.gather_distributed_history = (
        lambda local_history, history_base, context:
        gather_without_single_resume_duplication(
            original_gather, local_history, history_base, context
        )
    )
    try:
        diagnosis.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            receipt = {
                "stage": "state8_diagnosis_exact_resume_history_v1",
                "status": "complete",
                "single_device_resume_history_deduplication": "exact_prefix_only_v1",
                "distributed_history_semantics_changed": False,
            }
            (output_dir / "exact_resume_history.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        trainer.gather_distributed_history = original_gather


if __name__ == "__main__":
    main()
