#!/usr/bin/env python3
"""Build the full train+validation mix without T4 or proxy T6 rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


STATE_TOKEN = "<STUDENT_STATE>"
TASK_ALIASES = {
    "malrule_next_correctness": "predict_correctness",
    "malrule_next_response": "predict_response",
    "malrule_identity": "diagnose_misconception",
}
TRAIN_TASKS = {
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "predict_prospective_error",
    "predict_error_persistence",
}


def source_files(processed_dir: Path) -> list[tuple[str, str, Path]]:
    return [
        ("dbe_kt22", "train", processed_dir / "dbe_kt22_t1_t6/train.jsonl"),
        ("dbe_kt22", "valid", processed_dir / "dbe_kt22_t1_t6/valid.jsonl"),
        ("malrule", "train", processed_dir / "malrule_c1_clean/train.jsonl"),
        ("malrule", "valid", processed_dir / "malrule_c1_clean/val.jsonl"),
        ("eedi_mining", "train", processed_dir / "eedi_mining_c1_ir/train.jsonl"),
        ("eedi_response", "train", processed_dir / "eedi_response_t1_t2/train.jsonl"),
        ("eedi_response", "valid", processed_dir / "eedi_response_t1_t2/valid.jsonl"),
        ("xes3g5m", "train", processed_dir / "xes3g5m_t1_t6/train.jsonl"),
        ("xes3g5m", "valid", processed_dir / "xes3g5m_t1_t6/valid.jsonl"),
        ("foundationalassist", "train", processed_dir / "foundationalassist_core_tasks/train.jsonl"),
        ("foundationalassist", "valid", processed_dir / "foundationalassist_core_tasks/valid.jsonl"),
    ]


def canonicalize(row: dict[str, Any], source: str, original_split: str) -> dict[str, Any] | None:
    original_task = str(row.get("task") or "")
    task = TASK_ALIASES.get(original_task, original_task)
    if task not in TRAIN_TASKS:
        return None
    if source == "xes3g5m" and task == "predict_error_persistence":
        return None
    result = dict(row)
    original_id = str(result["example_id"])
    result["example_id"] = f"fullmix:{source}:{original_split}:{original_id}"
    result["dataset"] = source
    result["split"] = "train"
    result["task"] = task
    result["student_id"] = f"{source}::{result.get('student_id', '')}"
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "mix_source": source,
            "original_example_id": original_id,
            "original_split": original_split,
            "original_task": original_task,
            "training_protocol": "train_plus_validation_no_t4_strict_t6_only",
        }
    )
    result["metadata"] = metadata
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    eos_length = int(tokenizer.eos_token_id is not None)
    counts = Counter()
    seen_ids: set[str] = set()
    pending: list[dict[str, Any]] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def flush(output_handle) -> None:
        if not pending:
            return
        prompts = [str(row["prompt"]).replace(STATE_TOKEN, args.state_token_alias) for row in pending]
        prompt_lengths = tokenizer(prompts, add_special_tokens=False, return_length=True)["length"]
        targets = [str(row["target"]) for row in pending]
        target_lengths = tokenizer(targets, add_special_tokens=False, return_length=True)["length"]
        for row, prompt_length, target_length in zip(pending, prompt_lengths, target_lengths):
            source = str(row["metadata"]["mix_source"])
            task = str(row["task"])
            if prompt_length + target_length + eos_length > args.max_length:
                counts[f"dropped_length/{task}/{source}"] += 1
                continue
            example_id = str(row["example_id"])
            if example_id in seen_ids:
                raise ValueError(f"Duplicate example_id: {example_id}")
            seen_ids.add(example_id)
            output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts["kept"] += 1
            counts[f"kept_task/{task}"] += 1
            counts[f"kept_source/{source}"] += 1
            counts[f"kept_task_source/{task}/{source}"] += 1
            counts[f"kept_original_split/{row['metadata']['original_split']}"] += 1
        pending.clear()

    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        for source, original_split, path in source_files(args.processed_dir):
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as input_handle:
                for line in input_handle:
                    if not line.strip():
                        continue
                    counts["input_rows"] += 1
                    row = canonicalize(json.loads(line), source, original_split)
                    if row is None:
                        counts["excluded_rows"] += 1
                        continue
                    counts["eligible_rows"] += 1
                    pending.append(row)
                    if len(pending) >= args.batch_size:
                        flush(output_handle)
        flush(output_handle)

    stats = {
        "output_jsonl": str(args.output_jsonl),
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "state_token_alias": args.state_token_alias,
        "included_tasks": sorted(TRAIN_TASKS),
        "excluded_protocols": [
            "all estimate_mastery rows",
            "XES3G5M repeat-incorrectness proxy rows",
        ],
        "validation_policy": "all available validation rows are relabeled and merged into train",
        "counts": dict(sorted(counts.items())),
        "unique_example_ids": len(seen_ids),
    }
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
