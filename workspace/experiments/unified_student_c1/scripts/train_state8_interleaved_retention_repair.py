#!/usr/bin/env python3
"""State8 source-anchored repair with an audited materialized row order.

The ordinary repair trainer shuffles real rows after an index manifest has
materialized them.  That is desirable for normal training, but it defeats the
interleaved-retention experiment: early checkpoints must already contain T1,
T2, T3, T5, and balanced persistent/transient T6 support.  This wrapper keeps
the established source-anchored/causal/early-label objective unchanged and
only replaces the real-row permutation by the immutable JSONL order.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_diagnosis_first_source_anchored_repair as source_anchored


STAGE = "state8_interleaved_retention_optimizer_order_v1"
CHECKPOINT_PREFIXES = (20, 100, 240, 500, 1000)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity_real_schedule(
    original: Callable[[int, int, int], tuple[list[int], dict[int, int]]],
    real_count: int,
    causal_count: int,
    seed: int,
) -> tuple[list[int], dict[int, int]]:
    """Preserve only the real-row order; retain the frozen causal schedule."""
    _real_order, causal_schedule = original(real_count, causal_count, seed)
    return list(range(real_count)), causal_schedule


def t6_label(target: str) -> str:
    value = str(target).strip()
    if value == "<persistence>persistent</persistence>":
        return "persistent"
    if value == "<persistence>transient</persistence>":
        return "transient"
    raise ValueError(f"Unexpected T6 target: {target!r}")


def prefix_audit(rows: list[dict[str, Any]], stops: tuple[int, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stop in stops:
        if stop > len(rows):
            continue
        prefix = rows[:stop]
        tasks = Counter(str(row["task"]) for row in prefix)
        labels = Counter(
            t6_label(str(row["target"]))
            for row in prefix
            if row["task"] == "predict_error_persistence"
        )
        result[str(stop)] = {
            "task_counts": dict(sorted(tasks.items())),
            "t6_label_counts": dict(sorted(labels.items())),
        }
    return result


def audit_optimizer_order(
    input_rows: list[dict[str, Any]], history: list[dict[str, Any]]
) -> dict[str, Any]:
    if not input_rows or len(history) != len(input_rows):
        raise ValueError(
            f"Interleaved history cardinality drift: input={len(input_rows)} "
            f"history={len(history)}"
        )
    expected_ids = [str(row["example_id"]) for row in input_rows]
    observed_ids = [str(row.get("example_id") or "") for row in history]
    if observed_ids != expected_ids:
        mismatch = next(
            index
            for index, (expected, observed) in enumerate(
                zip(expected_ids, observed_ids, strict=True)
            )
            if expected != observed
        )
        raise ValueError(
            "Optimizer order differs from materialized order at "
            f"position={mismatch}: expected={expected_ids[mismatch]!r} "
            f"observed={observed_ids[mismatch]!r}"
        )
    if [int(row.get("global_position", -1)) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("Interleaved optimizer history coordinates are not contiguous")
    expected_prefix = prefix_audit(input_rows, CHECKPOINT_PREFIXES)
    history_rows = [
        {
            "task": str(row["task"]),
            "target": str(input_rows[index]["target"]),
        }
        for index, row in enumerate(history)
    ]
    observed_prefix = prefix_audit(history_rows, CHECKPOINT_PREFIXES)
    if observed_prefix != expected_prefix:
        raise ValueError("Interleaved optimizer prefix audit drift")
    return {
        "stage": STAGE,
        "status": "complete",
        "rows": len(history),
        "materialized_order_applied": True,
        "optimizer_order_exact": True,
        "example_ids_sha256": hashlib.sha256(
            "\n".join(observed_ids).encode()
        ).hexdigest(),
        "prefix_audit": observed_prefix,
        "raw_history_visible_to_decoder": False,
        "final_inference_route": "strict_state_only",
    }


def cli_path(flag: str) -> Path:
    if flag not in sys.argv or sys.argv.index(flag) + 1 >= len(sys.argv):
        raise ValueError(f"Missing required path: {flag}")
    return Path(sys.argv[sys.argv.index(flag) + 1]).resolve()


def main() -> None:
    output_dir = cli_path("--output-dir")
    train_path = cli_path("--train-jsonl")
    original = trainer.build_epoch_schedule

    def schedule(
        real_count: int, causal_count: int, seed: int
    ) -> tuple[list[int], dict[int, int]]:
        return identity_real_schedule(original, real_count, causal_count, seed)

    trainer.build_epoch_schedule = schedule
    try:
        source_anchored.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
            return
        rows = read_jsonl(train_path)
        history = json.loads(
            (output_dir / "trainer_history.json").read_text(encoding="utf-8")
        )
        receipt = audit_optimizer_order(rows, history)
        receipt.update(
            state_token_count=8,
            source_anchor_composed=True,
            causal_teacher_binding_composed=(
                float(os.environ.get("STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT", "0"))
                > 0
            ),
            early_label_binding_composed=(
                float(os.environ.get("STATE8_EARLY_LABEL_BINDING_WEIGHT", "0"))
                > 0
            ),
            entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        (output_dir / "interleaved_retention_repair.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        trainer.build_epoch_schedule = original


if __name__ == "__main__":
    main()
