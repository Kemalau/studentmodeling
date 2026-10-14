#!/usr/bin/env python3
"""Single-A100 topology sensitivity for the full-Eedi selective-KD arm.

Four serial micro-batches reproduce one four-rank optimizer batch.  The exact
diagnostic manifest order and the matching slice of the frozen full-epoch
causal schedule are retained.  This is a scheduling hedge, not a claim of
bitwise equivalence to rank-parallel RNG and reduction order.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import train_recurrent_student_m4 as trainer
import train_recurrent_student_m4_open_t3_selective_kd as selective


FULL_REAL_ROWS = 259_845
SEGMENT_START = 12_000


def cli_path(name: str) -> Path:
    if name not in sys.argv:
        raise ValueError(f"Missing required argument {name}")
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} has no value")
    return Path(sys.argv[index + 1]).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    train_path = cli_path("--train-jsonl")
    output_dir = cli_path("--output-dir")
    manifest_path = Path(os.environ["MANIFEST"]).resolve()
    expected_manifest_hash = os.environ["MANIFEST_SHA256"]
    if sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("G1 hedge diagnostic manifest SHA256 drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "layerwise_task_retention_index_manifest_v1":
        raise ValueError("G1 hedge manifest stage drift")
    if manifest.get("branch") != "eedi_full":
        raise ValueError("G1 hedge requires the full-Eedi manifest")
    indices = [int(value) for value in manifest.get("indices") or []]
    if len(indices) != 5000 or len(set(indices)) != 5000:
        raise ValueError("G1 hedge requires 5000 unique manifest indices")
    if int(manifest.get("start_global_position", -1)) != SEGMENT_START:
        raise ValueError("G1 hedge manifest start drift")

    original_read = trainer.read_jsonl
    original_epoch_schedule = trainer.build_epoch_schedule
    original_repeated_schedule = trainer.build_repeated_causal_schedule
    original_objective = trainer.backward_training_objective
    audit: dict[str, object] = {}

    def read_manifest_segment(path):
        rows = original_read(path)
        if Path(path).resolve() != train_path:
            return rows
        if sha256(train_path) != str(manifest.get("source_train_sha256")):
            raise ValueError("G1 hedge source train SHA256 drift")
        if max(indices) >= len(rows):
            raise ValueError("G1 hedge manifest index exceeds train corpus")
        selected = [rows[index] for index in indices]
        if len({str(row["example_id"]) for row in selected}) != len(selected):
            raise ValueError("G1 hedge selected duplicate examples")
        audit.update(
            selected_rows=len(selected),
            selected_example_ids_sha256=hashlib.sha256(
                "\n".join(str(row["example_id"]) for row in selected).encode()
            ).hexdigest(),
        )
        return selected

    def identity_epoch_schedule(real_count: int, causal_count: int, seed: int):
        if real_count != len(indices):
            raise ValueError("G1 hedge train support drift")
        _order, schedule = original_epoch_schedule(real_count, causal_count, seed)
        return list(range(real_count)), schedule

    def sliced_repeated_schedule(
        real_count: int, causal_count: int, fraction: float, seed: int
    ):
        if real_count != len(indices):
            raise ValueError("G1 hedge repeated-schedule support drift")
        full = original_repeated_schedule(FULL_REAL_ROWS, causal_count, fraction, seed)
        segment = {
            position - SEGMENT_START: pair
            for position, pair in full.items()
            if SEGMENT_START <= position < SEGMENT_START + real_count
        }
        if len(segment) != int(os.environ.get("EXPECTED_CAUSAL_PAIRS", "250")):
            raise ValueError(f"G1 hedge causal schedule support drift: {len(segment)}")
        audit["causal_schedule_sha256"] = hashlib.sha256(
            "\n".join(f"{key}:{segment[key]}" for key in sorted(segment)).encode()
        ).hexdigest()
        return segment

    def objective_with_global_progress(
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
        local_position = round(float(training_progress) * max(len(indices) - 1, 1))
        full_progress = (SEGMENT_START + local_position) / (FULL_REAL_ROWS - 1)
        return original_objective(
            wrapper,
            tokenizer,
            row,
            causal_row,
            cache,
            args,
            device,
            state_token_id,
            loss_scale,
            full_progress,
        )

    trainer.read_jsonl = read_manifest_segment
    trainer.build_epoch_schedule = identity_epoch_schedule
    trainer.build_repeated_causal_schedule = sliced_repeated_schedule
    trainer.backward_training_objective = objective_with_global_progress
    selective.main()
    receipt = {
        "stage": "layerwise_full_eedi_g1_topology_sensitivity_v1",
        "diagnostic_only": True,
        "topology": "single_a100_serial_accumulation_4",
        "source_topology": "four_rank_replicated_data_parallel",
        "bitwise_equivalence_claimed": False,
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_hash,
        "full_epoch_rows_for_schedule": FULL_REAL_ROWS,
        "source_global_position_start": SEGMENT_START,
        "source_global_position_end": SEGMENT_START + len(indices) - 1,
        "optimizer_batch_examples": 4,
        **audit,
    }
    (output_dir / "g1_topology_sensitivity_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
