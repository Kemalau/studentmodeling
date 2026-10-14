#!/usr/bin/env python3
"""Diagnosis-first State8 curriculum with a final state-only Eedi C4 phase.

The parent curriculum already implements the five equal history-dropout
quintiles (0, .25, .5, .75, 1).  This wrapper keeps that recipe unchanged but
moves every paired causal exposure into the final quintile.  Consequently C4
cannot be optimized while the answer path still has teacher/history support.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import train_recurrent_student_m4 as trainer
import train_state8_diagnosis_first_matched_teacher_curriculum as matched
from build_state8_eedi_same_question_causal_twins import build_twins


CAUSAL_START_FRACTION = 0.8


def cli_path(flag: str) -> Path:
    try:
        index = sys.argv.index(flag)
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Missing required CLI path: {flag}") from error


def replace_cli_path(flag: str, path: Path) -> None:
    try:
        sys.argv[sys.argv.index(flag) + 1] = str(path)
    except (ValueError, IndexError) as error:
        raise ValueError(f"Missing required CLI path: {flag}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_eedi_causal_data(output_dir: Path) -> dict[str, Any]:
    """Merge disjoint same-question Eedi twins into the parent C4 data."""
    train_path = cli_path("--train-jsonl")
    valid_path = cli_path("--valid-jsonl")
    causal_train_path = cli_path("--causal-train-jsonl")
    causal_valid_path = cli_path("--causal-valid-jsonl")
    source_rows = read_jsonl(train_path) + read_jsonl(valid_path)
    eedi_train, train_receipt = build_twins(
        source_rows, twins=100, seed=20260812, offset=0
    )
    eedi_valid, valid_receipt = build_twins(
        source_rows, twins=25, seed=20260812, offset=100
    )
    # The reusable builder numbers each independently materialized file from
    # zero.  Namespace IDs here so train/validation provenance remains globally
    # unique even though their selected question ranges are disjoint.
    for split, rows in (("train", eedi_train), ("valid", eedi_valid)):
        for row in rows:
            row["pair_id"] = f"state8-eedi-curriculum:{split}:{row['pair_id']}"
    base_train = read_jsonl(causal_train_path)
    base_valid = read_jsonl(causal_valid_path)
    merged_train = base_train + eedi_train
    merged_valid = base_valid + eedi_valid
    pair_ids = [str(row["pair_id"]) for row in merged_train + merged_valid]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Eedi curriculum causal pair IDs overlap")
    root = Path("/tmp/state8_diagnosis_first_eedi_curriculum_causal_v1")
    merged_train_path = root / "train.jsonl"
    merged_valid_path = root / "valid.jsonl"
    write_jsonl(merged_train_path, merged_train)
    write_jsonl(merged_valid_path, merged_valid)
    write_jsonl(output_dir / "eedi_same_question_valid.jsonl", eedi_valid)
    replace_cli_path("--causal-train-jsonl", merged_train_path)
    replace_cli_path("--causal-valid-jsonl", merged_valid_path)
    receipt = {
        "stage": "state8_diagnosis_first_eedi_causal_merge_v1",
        "status": "passed",
        "same_question_query_identical": True,
        "student_response_visible_to_decoder": False,
        "raw_history_visible_to_decoder": False,
        "canonical_label_role": "auxiliary_supervision_and_reporting",
        "base_train_pairs": len(base_train),
        "base_valid_pairs": len(base_valid),
        "eedi_train_pairs": len(eedi_train),
        "eedi_valid_pairs": len(eedi_valid),
        "eedi_same_question_valid_sha256": file_sha256(
            output_dir / "eedi_same_question_valid.jsonl"
        ),
        "merged_train_pairs": len(merged_train),
        "merged_valid_pairs": len(merged_valid),
        "source_counts_train": dict(
            sorted(Counter(str(row.get("source") or "malrule") for row in merged_train).items())
        ),
        "source_counts_valid": dict(
            sorted(Counter(str(row.get("source") or "malrule") for row in merged_valid).items())
        ),
        "train_sha256": file_sha256(merged_train_path),
        "valid_sha256": file_sha256(merged_valid_path),
        "eedi_train_build": train_receipt,
        "eedi_valid_build": valid_receipt,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eedi_causal_merge.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def shifted_repeated_causal_schedule(
    original,
    real_count: int,
    causal_count: int,
    fraction: float,
    seed: int,
) -> dict[int, int]:
    """Schedule every causal pair in the final state-only quintile."""
    start = round(real_count * CAUSAL_START_FRACTION)
    active_count = real_count - start
    if causal_count > active_count:
        raise ValueError(
            "Final state-only quintile cannot cover every causal pair: "
            f"pairs={causal_count} active_rows={active_count}"
        )
    schedule = original(active_count, causal_count, fraction, seed)
    shifted = {start + position: index for position, index in schedule.items()}
    if shifted and min(shifted) < start:
        raise AssertionError("Causal schedule escaped the state-only quintile")
    return shifted


def main() -> None:
    output_dir = cli_path("--output-dir")
    merge_receipt = materialize_eedi_causal_data(output_dir)
    original_schedule = trainer.build_repeated_causal_schedule

    def schedule(
        real_count: int, causal_count: int, fraction: float, seed: int
    ) -> dict[int, int]:
        return shifted_repeated_causal_schedule(
            original_schedule,
            real_count,
            causal_count,
            fraction,
            seed,
        )

    trainer.build_repeated_causal_schedule = schedule
    try:
        matched.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
            return
        history: list[dict[str, Any]] = json.loads(
            (output_dir / "trainer_history.json").read_text(encoding="utf-8")
        )
        completion = json.loads(
            (output_dir / "completion.json").read_text(encoding="utf-8")
        )
        causal = [row for row in history if row.get("causal_pair_id") is not None]
        start_position = round(len(history) * CAUSAL_START_FRACTION)
        if not causal:
            raise RuntimeError("Final state-only consolidation saw no causal pairs")
        if min(int(row["global_position"]) for row in causal) < start_position:
            raise RuntimeError("A causal pair was scheduled before the final quintile")
        if any(float(row["history_dropout"]) != 1.0 for row in causal):
            raise RuntimeError("Causal consolidation was not strictly state-only")
        if completion.get("all_causal_pairs_covered") is not True:
            raise RuntimeError("Final state-only consolidation missed causal pairs")
        receipt = {
            "stage": "state8_diagnosis_first_eedi_curriculum_v1",
            "status": "complete",
            "state_token_count": 8,
            "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
            "causal_consolidation_start_fraction": CAUSAL_START_FRACTION,
            "causal_consolidation_route": "final_state_only_quintile",
            "causal_exposures": len(causal),
            "eedi_causal_merge": merge_receipt,
            "first_causal_global_position": min(
                int(row["global_position"]) for row in causal
            ),
            "last_causal_global_position": max(
                int(row["global_position"]) for row in causal
            ),
            "raw_history_visible_to_decoder": False,
            "final_inference_route": "strict_state_only",
            "semantic_judge_authoritative": True,
            "exact_match_authoritative": False,
            "canonical_label_role": "auxiliary_supervision_and_reporting",
            "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (output_dir / "eedi_curriculum.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        trainer.build_repeated_causal_schedule = original_schedule


if __name__ == "__main__":
    main()
