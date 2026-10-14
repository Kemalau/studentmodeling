#!/usr/bin/env python3
"""Add atomic remote checkpoint/resume to the frozen recurrent T3 evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import evaluate_recurrent_formal_protocols as evaluator
from t3_resumable_progress import atomic_write_jsonl, load_progress


def main() -> None:
    original_parse = argparse.ArgumentParser.parse_args
    original_generation = evaluator.generation_records
    progress: dict[str, dict] = {}
    checkpoint_path: Path | None = None

    def parse(parser, *args, **kwargs):
        nonlocal progress, checkpoint_path
        parser.add_argument("--generation-progress-dir", type=Path, required=True)
        parser.add_argument("--generation-progress-key", required=True)
        namespace = original_parse(parser, *args, **kwargs)
        namespace.generation_progress_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = namespace.generation_progress_dir / "records.jsonl"
        return namespace

    def generation_records(
        wrapper,
        tokenizer,
        rows,
        states,
        alias,
        max_length,
        max_new_tokens,
    ):
        nonlocal progress
        if checkpoint_path is None:
            raise AssertionError("Resumable T3 progress path was not initialized")
        # Parse the two wrapper-only flags directly because the base evaluator
        # owns argument parsing and calls this function later.
        progress_dir = Path(sys.argv[sys.argv.index("--generation-progress-dir") + 1])
        progress_key = sys.argv[sys.argv.index("--generation-progress-key") + 1]
        reference_rows = []
        for row in rows:
            metadata = evaluator.evaluation_metadata(row)
            reference_rows.append(
                {
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "source": metadata["source"],
                    "protocol": metadata["protocol"],
                    "delta": metadata["delta"],
                    "target": row["target"],
                }
            )
        progress = load_progress(progress_dir, reference_rows, progress_key)
        records: list[dict] = []
        for index, (row, state_cpu, reference) in enumerate(
            zip(rows, states, reference_rows, strict=True), start=1
        ):
            example_id = str(row["example_id"])
            if example_id in progress:
                record = progress[example_id]
            else:
                state = (
                    None
                    if state_cpu is None
                    else state_cpu.unsqueeze(0).to(next(wrapper.parameters()).device)
                )
                output = evaluator.generate_one(
                    wrapper,
                    tokenizer,
                    state,
                    evaluator.closed_decode_text(row),
                    alias,
                    max_length,
                    max_new_tokens,
                )
                record = {**reference, "raw_output": output}
            records.append(record)
            atomic_write_jsonl(checkpoint_path, records)
            if index == 1 or index % 10 == 0:
                print(
                    {
                        "generated": index,
                        "total": len(rows),
                        "resumed_at_start": len(progress),
                    },
                    flush=True,
                )
        return records

    argparse.ArgumentParser.parse_args = parse
    evaluator.generation_records = generation_records
    try:
        evaluator.main()
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        evaluator.generation_records = original_generation


if __name__ == "__main__":
    main()
