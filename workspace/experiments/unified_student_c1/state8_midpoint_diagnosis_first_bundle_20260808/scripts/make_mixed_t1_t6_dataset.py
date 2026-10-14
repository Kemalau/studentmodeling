#!/usr/bin/env python3
"""Build a task- and source-balanced T1-T6 mixture from the selected datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from student_task_utils import CANONICAL_TASKS, write_jsonl


TASK_ALIASES = {
    "next_correctness": "predict_correctness",
    "malrule_next_correctness": "predict_correctness",
    "next_response": "predict_response",
    "malrule_next_response": "predict_response",
    "future_mastery": "estimate_mastery",
    "malrule_identity": "diagnose_misconception",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def canonicalize(row: dict[str, Any], source: str, split: str) -> dict[str, Any]:
    result = dict(row)
    original_task = str(result.get("task") or "")
    original_split = str(result.get("split") or "")
    result["task"] = TASK_ALIASES.get(original_task, original_task)
    result["split"] = split
    result["student_id"] = f"{source}::{result.get('student_id', '')}"
    metadata = dict(result.get("metadata") or {})
    metadata.update({"mix_source": source, "original_task": original_task, "original_split": original_split})
    result["metadata"] = metadata
    return result


def source_files(args: argparse.Namespace, split: str) -> list[tuple[str, Path]]:
    malrule_file = "train.jsonl" if split == "train" else "test_cross_template.jsonl"
    return [
        ("dbe_kt22", args.dbe_dir / f"{split}.jsonl"),
        ("xes3g5m", args.xes_dir / f"{split}.jsonl"),
        ("eedi_response", args.eedi_response_dir / f"{split}.jsonl"),
        ("malrule", args.malrule_dir / malrule_file),
        ("eedi_mining", args.eedi_mining_dir / f"{split}.jsonl"),
    ]


def load_split(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    rows = []
    for source, path in source_files(args, split):
        if not path.exists():
            raise FileNotFoundError(f"Missing {source} {split} data: {path}")
        rows.extend(canonicalize(row, source, split) for row in read_jsonl(path))
    unknown = sorted({row["task"] for row in rows} - set(CANONICAL_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks after canonicalization: {unknown}")
    return rows


def filter_by_length(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    state_token_alias: str,
) -> tuple[list[dict[str, Any]], Counter[tuple[str, str]]]:
    kept = []
    dropped: Counter[tuple[str, str]] = Counter()
    batch_size = 256
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [str(row["prompt"]).replace("<STUDENT_STATE>", state_token_alias) for row in batch]
        prompt_lengths = tokenizer(prompts, add_special_tokens=False, return_length=True)["length"]
        for row, prompt_length in zip(batch, prompt_lengths):
            target_length = len(tokenizer(str(row["target"]), add_special_tokens=False)["input_ids"])
            eos_length = int(tokenizer.eos_token_id is not None)
            if prompt_length + target_length + eos_length <= max_length:
                kept.append(row)
            else:
                dropped[(row["task"], row["metadata"]["mix_source"])] += 1
    return kept, dropped


def sample_group(rows: list[dict[str, Any]], count: int, rng: random.Random, repeat: bool) -> list[dict[str, Any]]:
    pool = list(rows)
    rng.shuffle(pool)
    if len(pool) >= count:
        return pool[:count]
    if not repeat:
        return pool
    selected = []
    while len(selected) < count:
        cycle = list(pool)
        rng.shuffle(cycle)
        selected.extend(cycle[: count - len(selected)])
    return selected


def task_quotas(source_rows: dict[str, list[dict[str, Any]]], total: int, max_repeat: int) -> dict[str, int]:
    names = sorted(source_rows)
    base, remainder = divmod(total, len(names))
    quotas = {
        source: min(base + int(index < remainder), len(source_rows[source]) * max_repeat)
        for index, source in enumerate(names)
    }
    remaining = total - sum(quotas.values())
    while remaining:
        eligible = [source for source in names if quotas[source] < len(source_rows[source]) * max_repeat]
        if not eligible:
            raise ValueError(f"Cannot fill {total} rows without exceeding max_repeat={max_repeat}")
        for source in eligible:
            if not remaining:
                break
            quotas[source] += 1
            remaining -= 1
    return quotas


def balanced_train(
    rows: list[dict[str, Any]],
    rows_per_task: int,
    max_repeat: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_task_source: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task_source[row["task"]][row["metadata"]["mix_source"]].append(row)

    selected = []
    for task in CANONICAL_TASKS:
        sources = by_task_source.get(task, {})
        if not sources:
            raise ValueError(f"No training rows for required task {task}")
        for source, quota in task_quotas(sources, rows_per_task, max_repeat).items():
            selected.extend(sample_group(sources[source], quota, rng, repeat=True))

    mixed = []
    repeat_counts: Counter[str] = Counter()
    rng.shuffle(selected)
    for mix_index, row in enumerate(selected):
        result = dict(row)
        original_id = str(result["example_id"])
        repeat_index = repeat_counts[original_id]
        repeat_counts[original_id] += 1
        result["example_id"] = f"mix:train:{mix_index:07d}:{original_id}"
        metadata = dict(result["metadata"])
        metadata.update({"original_example_id": original_id, "mix_repeat_index": repeat_index})
        result["metadata"] = metadata
        mixed.append(result)
    return mixed


def balanced_test(rows: list[dict[str, Any]], max_per_task_source: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["metadata"]["mix_source"])].append(row)
    selected = []
    for group_rows in groups.values():
        selected.extend(sample_group(group_rows, max_per_task_source, rng, repeat=False))
    rng.shuffle(selected)
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "by_task": dict(Counter(row["task"] for row in rows)),
        "by_source": dict(Counter(row["metadata"]["mix_source"] for row in rows)),
        "by_task_source": {
            f"{task}/{source}": count
            for (task, source), count in sorted(
                Counter((row["task"], row["metadata"]["mix_source"]) for row in rows).items()
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbe-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/dbe_kt22_t1_t6"))
    parser.add_argument("--xes-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/xes3g5m_t1_t6"))
    parser.add_argument("--eedi-response-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_response_t1_t2"))
    parser.add_argument("--malrule-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/malrule_c1_clean"))
    parser.add_argument("--eedi-mining-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_mining_c1_ir"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/mixed_t1_t6_balanced"))
    parser.add_argument("--train-rows-per-task", type=int, default=20_000)
    parser.add_argument("--max-train-repeat", type=int, default=8)
    parser.add_argument("--test-max-per-task-source", type=int, default=2_000)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    raw_train = load_split(args, "train")
    raw_test = load_split(args, "test")
    length_dropped: dict[str, Counter[tuple[str, str]]] = {"train": Counter(), "test": Counter()}
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        raw_train, length_dropped["train"] = filter_by_length(
            raw_train, tokenizer, args.max_length, args.state_token_alias
        )
        raw_test, length_dropped["test"] = filter_by_length(
            raw_test, tokenizer, args.max_length, args.state_token_alias
        )
    train = balanced_train(raw_train, args.train_rows_per_task, args.max_train_repeat, rng)
    test = balanced_test(raw_test, args.test_max_per_task_source, rng)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "test.jsonl", test)
    stats = {
        "seed": args.seed,
        "train_rows_per_task": args.train_rows_per_task,
        "max_train_repeat": args.max_train_repeat,
        "test_max_per_task_source": args.test_max_per_task_source,
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "state_token_alias": args.state_token_alias,
        "length_filter_dropped": {
            split: {f"{task}/{source}": count for (task, source), count in sorted(counts.items())}
            for split, counts in length_dropped.items()
        },
        "train": summarize(train),
        "test": summarize(test),
    }
    (args.out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
