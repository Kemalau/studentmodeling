#!/usr/bin/env python3
"""Candidate-logprob evaluation with one cached prompt forward per row."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import DynamicCache

from evaluate_c1 import (
    candidate_text,
    encode_prompt_text,
    labels_for_row,
    load_model,
    read_jsonl,
    summarize,
    write_json,
)


def common_prefix(sequences: list[list[int]]) -> list[int]:
    if not sequences:
        return []
    limit = min(len(sequence) for sequence in sequences)
    length = 0
    while length < limit and all(sequence[length] == sequences[0][length] for sequence in sequences[1:]):
        length += 1
    return sequences[0][:length]


def common_suffix_length(sequences: list[list[int]]) -> int:
    if not sequences:
        return 0
    limit = min(len(sequence) for sequence in sequences)
    length = 0
    while length < limit and all(sequence[-length - 1] == sequences[0][-length - 1] for sequence in sequences[1:]):
        length += 1
    return length


def repeat_cache(cache: Any, repeats: int) -> Any:
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    repeated = tuple(
        (
            key.repeat_interleave(repeats, dim=0),
            value.repeat_interleave(repeats, dim=0),
        )
        for key, value in legacy
    )
    return DynamicCache.from_legacy_cache(repeated)


def select_cache(cache: Any, index: int) -> Any:
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    selected = tuple((key[index:index + 1], value[index:index + 1]) for key, value in legacy)
    return DynamicCache.from_legacy_cache(selected)


def index_cache(cache: Any, indices: torch.Tensor) -> Any:
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    selected = tuple(
        (key.index_select(0, indices), value.index_select(0, indices))
        for key, value in legacy
    )
    return DynamicCache.from_legacy_cache(selected)


def prepare_row(
    tokenizer,
    row: dict[str, Any],
    max_length: int,
    state_token_alias: str | None,
    label_only: bool,
) -> dict[str, Any]:
    labels = labels_for_row(row)
    candidate_ids = [tokenizer(candidate_text(row, label), add_special_tokens=False)["input_ids"] for label in labels]
    shared = common_prefix(candidate_ids)
    remainders = [ids[len(shared):] for ids in candidate_ids]
    if any(not remainder for remainder in remainders):
        shared = shared[:-1]
        remainders = [ids[len(shared):] for ids in candidate_ids]
    if label_only:
        suffix_length = common_suffix_length(remainders)
        if suffix_length and all(len(remainder) > suffix_length for remainder in remainders):
            remainders = [remainder[:-suffix_length] for remainder in remainders]
    prompt = encode_prompt_text(str(row["prompt"]), state_token_alias)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    longest_target = max(len(ids) for ids in candidate_ids)
    if len(prompt_ids) + longest_target > max_length:
        prompt_ids = prompt_ids[-(max_length - longest_target):]
    return {
        "labels": labels,
        "remainders": remainders,
        "base_ids": prompt_ids + shared,
    }


@torch.inference_mode()
def score_rows(
    tokenizer,
    model,
    rows: list[dict[str, Any]],
    max_length: int,
    state_token_alias: str | None,
    label_only: bool,
) -> list[dict[str, float]]:
    device = getattr(model, "device", next(model.parameters()).device)
    prepared = [prepare_row(tokenizer, row, max_length, state_token_alias, label_only) for row in rows]
    padded_length = max(len(item["base_ids"]) for item in prepared)
    input_ids = torch.full(
        (len(prepared), padded_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, item in enumerate(prepared):
        length = len(item["base_ids"])
        input_ids[index, -length:] = torch.tensor(item["base_ids"], dtype=torch.long, device=device)
        attention_mask[index, -length:] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    base_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        logits_to_keep=1,
    )
    first_log_probs = torch.log_softmax(base_output.logits[:, -1].float(), dim=-1)
    batch_scores: list[list[float]] = []
    continuation_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_index, item in enumerate(prepared):
        remainders = item["remainders"]
        scores = [float(first_log_probs[row_index, remainder[0]].item()) for remainder in remainders]
        for candidate_index, remainder in enumerate(remainders):
            if len(remainder) > 1:
                continuation_groups[len(remainder)].append((row_index, candidate_index))
        batch_scores.append(scores)
    for length, jobs in continuation_groups.items():
        row_indices = torch.tensor([row_index for row_index, _ in jobs], dtype=torch.long, device=device)
        continuation = torch.tensor(
            [prepared[row_index]["remainders"][candidate_index][:-1] for row_index, candidate_index in jobs],
            dtype=torch.long,
            device=device,
        )
        cache = index_cache(base_output.past_key_values, row_indices)
        prefix_attention = attention_mask.index_select(0, row_indices)
        attention = torch.cat(
            [
                prefix_attention,
                torch.ones((len(jobs), length - 1), dtype=torch.long, device=device),
            ],
            dim=1,
        )
        offsets = torch.arange(length - 1, dtype=torch.long, device=device).unsqueeze(0)
        base_lengths = torch.tensor(
            [len(prepared[row_index]["base_ids"]) for row_index, _ in jobs],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        continuation_positions = base_lengths + offsets
        cache_positions = torch.arange(
            padded_length,
            padded_length + length - 1,
            dtype=torch.long,
            device=device,
        )
        output = model(
            input_ids=continuation,
            attention_mask=attention,
            position_ids=continuation_positions,
            cache_position=cache_positions,
            past_key_values=cache,
            use_cache=False,
            logits_to_keep=continuation.shape[1],
        )
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
        targets = torch.tensor(
            [prepared[row_index]["remainders"][candidate_index][1:] for row_index, candidate_index in jobs],
            dtype=torch.long,
            device=device,
        )
        continuation_scores = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        for job_index, (row_index, candidate_index) in enumerate(jobs):
            batch_scores[row_index][candidate_index] += float(continuation_scores[job_index].item())
    return [
        {label: score for label, score in zip(item["labels"], scores)}
        for item, scores in zip(prepared, batch_scores)
    ]


@torch.inference_mode()
def score_row(tokenizer, model, row: dict[str, Any], max_length: int, state_token_alias: str | None) -> dict[str, float]:
    device = getattr(model, "device", next(model.parameters()).device)
    labels = labels_for_row(row)
    candidate_ids = [tokenizer(candidate_text(row, label), add_special_tokens=False)["input_ids"] for label in labels]
    shared = common_prefix(candidate_ids)
    remainders = [ids[len(shared):] for ids in candidate_ids]
    if any(not remainder for remainder in remainders):
        shared = shared[:-1]
        remainders = [ids[len(shared):] for ids in candidate_ids]

    prompt = encode_prompt_text(str(row["prompt"]), state_token_alias)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    longest_target = max(len(ids) for ids in candidate_ids)
    if len(prompt_ids) + longest_target > max_length:
        prompt_ids = prompt_ids[-(max_length - longest_target):]
    base_ids = prompt_ids + shared
    base_tensor = torch.tensor([base_ids], dtype=torch.long, device=device)
    base_attention = torch.ones_like(base_tensor)
    base_output = model(input_ids=base_tensor, attention_mask=base_attention, use_cache=True)
    first_log_probs = torch.log_softmax(base_output.logits[0, -1].float(), dim=-1)
    scores = [float(first_log_probs[remainder[0]].item()) for remainder in remainders]

    groups: dict[int, list[int]] = defaultdict(list)
    for index, remainder in enumerate(remainders):
        if len(remainder) > 1:
            groups[len(remainder)].append(index)
    for length, indices in groups.items():
        continuation = torch.tensor(
            [remainders[index][:-1] for index in indices],
            dtype=torch.long,
            device=device,
        )
        cache = repeat_cache(base_output.past_key_values, len(indices))
        attention = torch.ones(
            (len(indices), len(base_ids) + length - 1),
            dtype=torch.long,
            device=device,
        )
        output = model(
            input_ids=continuation,
            attention_mask=attention,
            past_key_values=cache,
            use_cache=False,
        )
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
        targets = torch.tensor(
            [remainders[index][1:] for index in indices],
            dtype=torch.long,
            device=device,
        )
        continuation_scores = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        for local_index, row_index in enumerate(indices):
            scores[row_index] += float(continuation_scores[local_index].item())
    return {label: score for label, score in zip(labels, scores)}


@torch.inference_mode()
def evaluate(
    model_path: str,
    adapter_path: str | None,
    rows: list[dict[str, Any]],
    max_length: int,
    state_token_alias: str | None,
    progress_every: int,
    prefill_batch_size: int,
    large_label_prefill_batch_size: int | None,
    label_only: bool,
) -> dict[str, Any]:
    tokenizer, model = load_model(model_path, adapter_path, load_4bit=True)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    result = {}
    completed = 0
    for task, task_rows in sorted(by_task.items()):
        task_rows = sorted(task_rows, key=lambda row: len(str(row["prompt"])))
        task_batch_size = prefill_batch_size
        if large_label_prefill_batch_size and max(len(labels_for_row(row)) for row in task_rows) > 8:
            task_batch_size = large_label_prefill_batch_size
        predictions = []
        probabilities = []
        print(f"[cached_eval] task={task} rows={len(task_rows)} batch={task_batch_size}", flush=True)
        for start in range(0, len(task_rows), task_batch_size):
            batch_rows = task_rows[start:start + task_batch_size]
            batch_scores = score_rows(tokenizer, model, batch_rows, max_length, state_token_alias, label_only)
            for row, scores in zip(batch_rows, batch_scores):
                labels = labels_for_row(row)
                values = torch.tensor([scores[label] for label in labels], dtype=torch.float32)
                normalized = torch.softmax(values, dim=0)
                probability = {label: float(normalized[position].item()) for position, label in enumerate(labels)}
                probabilities.append(probability)
                predictions.append(max(probability, key=probability.get))
            completed += len(batch_rows)
            index = start + len(batch_rows)
            if progress_every and (start == 0 or index % progress_every == 0 or index == len(task_rows)):
                print(f"[cached_eval] task={task} row={index}/{len(task_rows)} total={completed}", flush=True)
        task_result = summarize(task_rows, predictions, probabilities)
        source_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(task_rows):
            source = str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")
            source_indices[source].append(index)
        task_result["by_source"] = {
            source: summarize(
                [task_rows[index] for index in indices],
                [predictions[index] for index in indices],
                [probabilities[index] for index in indices],
            )
            for source, indices in sorted(source_indices.items())
        }
        task_result["scoring_mode"] = (
            "candidate_label_logprob_cached_prefix" if label_only
            else "candidate_target_logprob_cached_prefix"
        )
        result[task] = task_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--max-samples-per-task", type=int)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--state-token-alias")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    parser.add_argument("--large-label-prefill-batch-size", type=int)
    parser.add_argument("--label-only", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.eval_jsonl)
    if args.tasks:
        allowed = set(args.tasks)
        rows = [row for row in rows if row["task"] in allowed]
    if args.max_samples_per_task:
        counts: Counter[str] = Counter()
        kept = []
        for row in rows:
            if counts[row["task"]] < args.max_samples_per_task:
                kept.append(row)
                counts[row["task"]] += 1
        rows = kept
    result = {
        "eval_jsonl": str(args.eval_jsonl),
        "max_samples_per_task": args.max_samples_per_task,
        "model": {
            "model_path": args.model,
            "adapter_path": args.adapter,
            "metrics": evaluate(
                args.model,
                args.adapter,
                rows,
                args.max_length,
                args.state_token_alias,
                args.progress_every,
                args.prefill_batch_size,
                args.large_label_prefill_batch_size,
                args.label_only,
            ),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    write_json(args.output_json, result)


if __name__ == "__main__":
    main()
