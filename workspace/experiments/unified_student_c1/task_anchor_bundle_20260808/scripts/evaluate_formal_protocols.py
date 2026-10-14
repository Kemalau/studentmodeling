#!/usr/bin/env python3
"""Evaluate formal student-modeling protocols with classification and generation."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from evaluate_c1 import (
    BOTTLENECK_STYLES,
    build_bottleneck_attention_mask,
    encode_candidate,
    load_model,
    locate_bottleneck_positions,
)
from evaluate_c1_cached import score_rows
from t3_llm_judge import config_from_args, judge_t3_records, write_audit_sample
from t3_judge_utils import natural_schema_valid


STATE_TOKEN = "<STUDENT_STATE>"
BINARY_TASKS = {"predict_correctness", "predict_error_persistence"}
GENERATION_TASKS = {"predict_response", "diagnose_misconception", "predict_prospective_error"}
TAG_BY_TASK = {
    "predict_response": "predicted_response",
    "diagnose_misconception": "misconception",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    # Recurrent formal manifests preserve the frozen protocol under
    # ``original_metadata`` after adding audited update/decode views.
    metadata = row.get("metadata") or row.get("original_metadata") or {}
    delta = str(metadata.get("prospective_delta") or "")
    return (
        str(row["task"]),
        str(metadata.get("eval_source") or row.get("dataset") or "unknown"),
        str(metadata.get("eval_protocol") or "unknown"),
        delta,
    )


def stable_rank(row: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row['example_id']}".encode("utf-8")).hexdigest()


def sample_rows(rows: list[dict[str, Any]], maximum: int | None, seed: int) -> list[dict[str, Any]]:
    if maximum is None:
        return rows
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    selected = []
    for key, group in sorted(grouped.items()):
        if key[0] in BINARY_TASKS:
            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group:
                by_label[str(row["target_label"])].append(row)
            labels = sorted(by_label)
            base, remainder = divmod(maximum, max(1, len(labels)))
            for index, label in enumerate(labels):
                label_rows = sorted(by_label[label], key=lambda row: stable_rank(row, seed))
                selected.extend(label_rows[: base + int(index < remainder)])
            if sum(min(len(by_label[label]), base + int(index < remainder)) for index, label in enumerate(labels)) < maximum:
                used = {row["example_id"] for row in selected}
                remaining = sorted(
                    (row for row in group if row["example_id"] not in used),
                    key=lambda row: stable_rank(row, seed),
                )
                selected.extend(remaining[: maximum - sum(row in selected for row in group)])
        else:
            selected.extend(sorted(group, key=lambda row: stable_rank(row, seed))[:maximum])
    return sorted(selected, key=lambda row: (group_key(row), stable_rank(row, seed)))


def evaluation_shard(
    rows: list[dict[str, Any]], shard_index: int | None, num_shards: int | None
) -> list[dict[str, Any]]:
    """Return one deterministic, disjoint shard of the filtered panel.

    Partitioning by sorted immutable example ID makes the assignment independent
    of input JSONL ordering.  Round-robin assignment keeps shard sizes within
    one row; the downstream merge rejects duplicates, gaps, and invariant drift.
    """
    if shard_index is None and num_shards is None:
        return rows
    if shard_index is None or num_shards is None:
        raise ValueError("--shard-index and --num-shards must be provided together")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Evaluation shard must satisfy 0 <= index < num_shards")
    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    if len({str(row["example_id"]) for row in ordered}) != len(ordered):
        raise ValueError("Evaluation rows contain duplicate example IDs")
    return [
        row
        for position, row in enumerate(ordered)
        if position % num_shards == shard_index
    ]


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.I | re.S)
    return html.unescape(match.group(1).strip()) if match else None


def normalize_text(value: str | None) -> str:
    text = html.unescape(str(value or "")).casefold().strip()
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    return re.sub(r"\s+", "", text)


def numeric_value(value: str | None):
    text = normalize_text(value).replace(",", "")
    if not text or len(text) > 80 or not re.fullmatch(r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+\-]?(?:\d+(?:\.\d+)?|\.\d+))?", text):
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def mathematically_equivalent(prediction: str | None, gold: str | None) -> bool:
    if normalize_text(prediction) == normalize_text(gold):
        return True
    pred_value = numeric_value(prediction)
    gold_value = numeric_value(gold)
    if pred_value is None or gold_value is None:
        return False
    return pred_value == gold_value


def binary_auroc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2
        rank_sum += average_rank * sum(label for _, label in pairs[index : end + 1])
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_auprc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(
        sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True),
        1,
    ):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def class_f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def binary_metrics(records: list[dict[str, Any]], positive: str) -> dict[str, Any]:
    gold = [int(record["gold"] == positive) for record in records]
    probability = [float(record["probabilities"].get(positive, 0.0)) for record in records]
    prediction = [int(record["prediction"] == positive) for record in records]
    true_positive = sum(label == 1 and pred == 1 for label, pred in zip(gold, prediction))
    false_positive = sum(label == 0 and pred == 1 for label, pred in zip(gold, prediction))
    false_negative = sum(label == 1 and pred == 0 for label, pred in zip(gold, prediction))
    true_negative = sum(label == 0 and pred == 0 for label, pred in zip(gold, prediction))
    positive_f1 = class_f1(true_positive, false_positive, false_negative)
    negative_f1 = class_f1(true_negative, false_negative, false_positive)
    result = {
        "n": len(records),
        "positive": positive,
        "positive_rate": sum(gold) / len(gold) if gold else None,
        "accuracy": sum(label == pred for label, pred in zip(gold, prediction)) / max(1, len(gold)),
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "positive_precision": true_positive / max(1, true_positive + false_positive),
        "positive_recall": true_positive / max(1, true_positive + false_negative),
        "nll": sum(
            -math.log(max(min(prob if label else 1 - prob, 1 - 1e-12), 1e-12))
            for label, prob in zip(gold, probability)
        ) / max(1, len(gold)),
        "brier": sum((prob - label) ** 2 for label, prob in zip(gold, probability)) / max(1, len(gold)),
    }
    result["auroc"] = binary_auroc(gold, probability)
    result["auprc"] = binary_auprc(gold, probability)
    return result


def generation_metrics(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    if task in TAG_BY_TASK:
        # T3 supports both the legacy <misconception> response and the natural
        # <label>/<diagnosis>/<evidence> schema.  Parse the schema used by each
        # reference row.  In particular, never let two missing legacy tags
        # compare equal: that previously produced a spurious exact-match 1.0
        # for otherwise valid natural-schema generations.
        natural_t3 = [
            task == "diagnose_misconception"
            and extract_tag(str(record.get("target") or ""), "diagnosis") is not None
            for record in records
        ]
        tags = [
            "diagnosis" if is_natural else TAG_BY_TASK[task]
            for is_natural in natural_t3
        ]
        parsed = [
            extract_tag(str(record.get("raw_output") or ""), tag)
            for record, tag in zip(records, tags)
        ]
        gold = [
            extract_tag(str(record.get("target") or ""), tag)
            for record, tag in zip(records, tags)
        ]
        exact = [
            prediction is not None
            and target is not None
            and normalize_text(prediction) == normalize_text(target)
            for prediction, target in zip(parsed, gold)
        ]
        result = {
            "n": len(records),
            "format_valid_rate": sum(value is not None for value in parsed) / max(1, len(records)),
            "normalized_exact_match": sum(exact) / max(1, len(records)),
        }
        if task == "predict_response":
            result["math_equivalence"] = sum(
                mathematically_equivalent(prediction, target) for prediction, target in zip(parsed, gold)
            ) / max(1, len(records))
        else:
            result["full_taxonomy_retrieval_pending"] = True
            result["natural_schema_valid_rate"] = sum(
                natural_schema_valid(str(record.get("raw_output") or ""))
                for record in records
            ) / max(1, len(records))
            if any(natural_t3):
                predicted_labels = [
                    extract_tag(str(record.get("raw_output") or ""), "label")
                    for record in records
                ]
                gold_labels = [
                    extract_tag(str(record.get("target") or ""), "label")
                    for record in records
                ]
                result["canonical_label_exact_match"] = sum(
                    prediction is not None
                    and target is not None
                    and normalize_text(prediction) == normalize_text(target)
                    for prediction, target in zip(predicted_labels, gold_labels)
                ) / max(1, len(records))
                result["exact_match_role"] = "auxiliary_only_not_authoritative"
            judged = [record for record in records if "llm_judge_correct" in record]
            if judged:
                result.update(
                    {
                        "llm_judge_accuracy": sum(bool(record["llm_judge_correct"]) for record in judged) / len(judged),
                        "llm_judge_n": len(judged),
                        "llm_judge_mean_confidence": sum(float(record["llm_judge_confidence"]) for record in judged)
                        / len(judged),
                        "llm_judge_exact_shortcuts": sum(
                            record.get("llm_judge_mode") == "exact_match_shortcut" for record in judged
                        ),
                        "llm_judge_api_or_cache": sum(
                            record.get("llm_judge_mode") in {"api", "cache"} for record in judged
                        ),
                    }
                )
        return result
    correctness = [extract_tag(record["raw_output"], "correctness") for record in records]
    response = [extract_tag(record["raw_output"], "predicted_response") for record in records]
    gold_correctness = [extract_tag(record["target"], "correctness") for record in records]
    gold_response = [extract_tag(record["target"], "predicted_response") for record in records]
    correctness_match = [
        normalize_text(prediction) == normalize_text(target)
        for prediction, target in zip(correctness, gold_correctness)
    ]
    response_match = [
        normalize_text(prediction) == normalize_text(target)
        for prediction, target in zip(response, gold_response)
    ]
    response_equivalence = [
        mathematically_equivalent(prediction, target) for prediction, target in zip(response, gold_response)
    ]
    return {
        "n": len(records),
        "format_valid_rate": sum(
            pred_correctness is not None and pred_response is not None
            for pred_correctness, pred_response in zip(correctness, response)
        ) / max(1, len(records)),
        "correctness_accuracy": sum(correctness_match) / max(1, len(records)),
        "response_normalized_exact_match": sum(response_match) / max(1, len(records)),
        "response_math_equivalence": sum(response_equivalence) / max(1, len(records)),
        "joint_exact_match": sum(a and b for a, b in zip(correctness_match, response_match)) / max(1, len(records)),
        "joint_math_equivalence": sum(a and b for a, b in zip(correctness_match, response_equivalence)) / max(1, len(records)),
    }


@torch.inference_mode()
def classify_rows(tokenizer, model, rows: list[dict[str, Any]], max_length: int, batch_size: int, state_alias: str):
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        scores = score_rows(tokenizer, model, batch, max_length, state_alias, label_only=True)
        for row, row_scores in zip(batch, scores):
            labels = list(row["candidate_labels"])
            values = torch.tensor([row_scores[label] for label in labels], dtype=torch.float32)
            normalized = torch.softmax(values, dim=0)
            probabilities = {label: float(normalized[index]) for index, label in enumerate(labels)}
            records.append(
                {
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "source": row["metadata"]["eval_source"],
                    "protocol": row["metadata"]["eval_protocol"],
                    "delta": row["metadata"].get("prospective_delta"),
                    "gold": str(row["target_label"]),
                    "prediction": max(probabilities, key=probabilities.get),
                    "probabilities": probabilities,
                }
            )
        print(f"[classify] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return records


@torch.inference_mode()
def classify_rows_bottleneck(
    tokenizer,
    model,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    state_alias: str,
    style: str,
):
    """Score binary candidates with the same hard mask used in training."""
    device = getattr(model, "device", next(model.parameters()).device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    items = []
    for row_index, row in enumerate(rows):
        for label in row["candidate_labels"]:
            item = encode_candidate(
                tokenizer,
                row,
                str(label),
                max_length,
                bottleneck_mask=True,
                state_token_alias=state_alias,
            )
            item["row_index"] = row_index
            items.append(item)
    scores: list[dict[str, float]] = [dict() for _ in rows]
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        padded_length = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention = []
        for item in batch:
            padding = padded_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * padding)
            attention.append([1] * len(item["input_ids"]) + [0] * padding)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_2d = torch.tensor(attention, dtype=torch.long, device=device)
        attention_4d = build_bottleneck_attention_mask(
            attention_2d,
            torch.tensor(
                [item["history_start"] for item in batch],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [item["state_pos"] for item in batch],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [item["prefix_len"] for item in batch],
                dtype=torch.long,
                device=device,
            ),
            style,
        )
        logits = model(input_ids=input_tensor, attention_mask=attention_4d).logits
        for batch_index, item in enumerate(batch):
            score = 0.0
            for position in range(item["prefix_len"], len(item["input_ids"])):
                log_probs = torch.log_softmax(
                    logits[batch_index, position - 1].float(), dim=-1
                )
                score += float(log_probs[item["input_ids"][position]].item())
            scores[item["row_index"]][str(item["label"])] = score
        print(
            f"[classify-masked] {min(start + batch_size, len(items))}/{len(items)}",
            flush=True,
        )
    records = []
    for row, row_scores in zip(rows, scores, strict=True):
        labels = [str(label) for label in row["candidate_labels"]]
        values = torch.tensor([row_scores[label] for label in labels])
        probabilities = dict(zip(labels, torch.softmax(values, dim=0).tolist()))
        metadata = row.get("metadata") or row.get("original_metadata") or {}
        records.append(
            {
                "example_id": row["example_id"],
                "task": row["task"],
                "source": metadata["eval_source"],
                "protocol": metadata["eval_protocol"],
                "delta": metadata.get("prospective_delta"),
                "gold": str(row["target_label"]),
                "prediction": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }
        )
    return records


@torch.inference_mode()
def generate_rows(
    tokenizer,
    model,
    rows: list[dict[str, Any]],
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    state_alias: str,
):
    tokenizer.padding_side = "left"
    records = []
    device = getattr(model, "device", next(model.parameters()).device)
    model.config.use_cache = True
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [str(row["prompt"]).replace(STATE_TOKEN, state_alias) for row in batch]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        generated = output[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(batch, texts):
            records.append(
                {
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "source": row["metadata"]["eval_source"],
                    "protocol": row["metadata"]["eval_protocol"],
                    "delta": row["metadata"].get("prospective_delta"),
                    "target": row["target"],
                    "raw_output": text.strip(),
                }
            )
        print(f"[generate] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return records


def prepare_bottleneck_prompt(
    tokenizer,
    raw_prompt: str,
    state_alias: str,
    max_length: int,
) -> tuple[list[int], int, int]:
    """Encode a prompt while retaining the history/STATE mask coordinates."""
    history_start, state_pos = locate_bottleneck_positions(
        tokenizer, raw_prompt, state_alias
    )
    prompt = raw_prompt.replace(STATE_TOKEN, state_alias)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_length:
        trim = len(prompt_ids) - max_length
        prompt_ids = prompt_ids[-max_length:]
        history_start -= trim
        state_pos -= trim
        if history_start < 0 or state_pos < 0:
            raise ValueError("Prompt truncation removed bottleneck coordinates")
    return prompt_ids, history_start, state_pos


def incremental_bottleneck_attention_mask(
    key_value_length: int,
    history_start: int,
    state_pos: int,
    device: torch.device,
) -> torch.Tensor:
    """Mask raw-history KV columns for one post-STATE decode query."""
    if not (0 <= history_start < state_pos < key_value_length):
        raise ValueError(
            "Invalid incremental mask coordinates: "
            f"history_start={history_start}, state_pos={state_pos}, "
            f"key_value_length={key_value_length}"
        )
    mask = torch.zeros(
        (1, 1, 1, key_value_length), dtype=torch.float32, device=device
    )
    mask[..., history_start:state_pos] = torch.finfo(torch.float32).min
    return mask


@torch.inference_mode()
def generate_one_bottleneck(
    tokenizer,
    model,
    raw_prompt: str,
    max_length: int,
    max_new_tokens: int,
    state_alias: str,
    style: str,
    stop_texts: tuple[str, ...] = (),
) -> str:
    """Greedy decode while keeping the hard history bottleneck at every step."""
    prompt_ids, history_start, state_pos = prepare_bottleneck_prompt(
        tokenizer, raw_prompt, state_alias, max_length
    )
    device = getattr(model, "device", next(model.parameters()).device)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_2d = torch.ones_like(input_ids)
    prompt_end = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)
    attention_4d = build_bottleneck_attention_mask(
        attention_2d,
        torch.tensor([history_start], dtype=torch.long, device=device),
        torch.tensor([state_pos], dtype=torch.long, device=device),
        prompt_end,
        style,
    )
    output = model(
        input_ids=input_ids,
        attention_mask=attention_4d,
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    token = int(output.logits[0, -1].argmax().item())
    generated: list[int] = []
    for _ in range(max_new_tokens):
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if stop_texts:
            partial = tokenizer.decode(generated, skip_special_tokens=True)
            if all(stop_text.casefold() in partial.casefold() for stop_text in stop_texts):
                break
        if len(prompt_ids) + len(generated) >= max_length:
            break
        past_length = len(prompt_ids) + len(generated) - 1
        key_value_length = past_length + 1
        next_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        step_mask = incremental_bottleneck_attention_mask(
            key_value_length, history_start, state_pos, device
        )
        position = torch.tensor([[past_length]], dtype=torch.long, device=device)
        output = model(
            input_ids=next_ids,
            attention_mask=step_mask,
            position_ids=position,
            cache_position=position.view(-1),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        token = int(output.logits[0, -1].argmax().item())
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_rows_bottleneck(
    tokenizer,
    model,
    rows: list[dict[str, Any]],
    max_length: int,
    max_new_tokens: int,
    state_alias: str,
    style: str,
):
    records = []
    for index, row in enumerate(rows, start=1):
        stop_texts = {
            "predict_response": ("</predicted_response>",),
            "diagnose_misconception": ("</misconception>",),
            "predict_prospective_error": (
                "</correctness>",
                "</predicted_response>",
            ),
        }.get(str(row["task"]), ())
        text = generate_one_bottleneck(
            tokenizer,
            model,
            str(row["prompt"]),
            max_length,
            max_new_tokens,
            state_alias,
            style,
            stop_texts,
        )
        metadata = row.get("metadata") or row.get("original_metadata") or {}
        records.append(
            {
                "example_id": row["example_id"],
                "task": row["task"],
                "source": metadata["eval_source"],
                "protocol": metadata["eval_protocol"],
                "delta": metadata.get("prospective_delta"),
                "target": row["target"],
                "raw_output": text,
            }
        )
        print(f"[generate-masked] {index}/{len(rows)}", flush=True)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["source"], record["protocol"], str(record.get("delta") or ""))].append(record)
    result = {}
    for (task, source, protocol, delta), group in sorted(grouped.items()):
        name = "/".join(part for part in (task, source, protocol, f"delta={delta}" if delta else "") if part)
        if task == "predict_correctness":
            result[name] = binary_metrics(group, "correct")
        elif task == "predict_error_persistence":
            result[name] = binary_metrics(group, "persistent")
        else:
            result[name] = generation_metrics(group, task)
    return result


def redacted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        item = {key: value for key, value in record.items() if key not in {"raw_output", "target"}}
        if "raw_output" in record:
            item["raw_output_sha256"] = hashlib.sha256(record["raw_output"].encode("utf-8")).hexdigest()
            item["raw_output_length"] = len(record["raw_output"])
            item["target_sha256"] = hashlib.sha256(record["target"].encode("utf-8")).hexdigest()
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-group", type=int)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--classification-batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--bottleneck-mask", action="store_true")
    parser.add_argument(
        "--bottleneck-style", choices=BOTTLENECK_STYLES, default="standard"
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-unredacted-predictions", action="store_true")
    parser.add_argument("--t3-llm-judge", action="store_true")
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--judge-base-url", default="http://1.13.190.7:25242")
    parser.add_argument("--judge-workers", type=int, default=12)
    parser.add_argument("--judge-cache", type=Path)
    parser.add_argument("--judge-audit-samples", type=int, default=100)
    args = parser.parse_args()

    rows = read_jsonl(args.eval_jsonl)
    if args.tasks:
        allowed = set(args.tasks)
        rows = [row for row in rows if row["task"] in allowed]
    if args.sources:
        allowed_sources = set(args.sources)
        rows = [row for row in rows if row.get("metadata", {}).get("eval_source") in allowed_sources]
    unsharded_rows = len(rows)
    rows = evaluation_shard(rows, args.shard_index, args.num_shards)
    rows = sample_rows(rows, args.max_samples_per_group, args.seed)
    classification_rows = [row for row in rows if row["task"] in BINARY_TASKS]
    generation_rows = [row for row in rows if row["task"] in GENERATION_TASKS]
    tokenizer, model = load_model(
        args.model,
        args.adapter,
        bottleneck_mask=args.bottleneck_mask,
        load_4bit=True,
    )
    if args.bottleneck_mask:
        classification_records = classify_rows_bottleneck(
            tokenizer,
            model,
            classification_rows,
            args.max_length,
            args.classification_batch_size,
            args.state_token_alias,
            args.bottleneck_style,
        )
    else:
        classification_records = classify_rows(
            tokenizer,
            model,
            classification_rows,
            args.max_length,
            args.classification_batch_size,
            args.state_token_alias,
        )
    if args.bottleneck_mask:
        generation_records = generate_rows_bottleneck(
            tokenizer,
            model,
            generation_rows,
            args.max_length,
            args.max_new_tokens,
            args.state_token_alias,
            args.bottleneck_style,
        )
    else:
        generation_records = generate_rows(
            tokenizer,
            model,
            generation_rows,
            args.max_length,
            args.max_new_tokens,
            args.generation_batch_size,
            args.state_token_alias,
        )
    records = classification_records + generation_records
    if args.t3_llm_judge:
        judge_config = config_from_args(args.judge_model, args.judge_base_url, args.judge_workers)
        cache_path = args.judge_cache or args.output_dir / "t3_judge_cache.jsonl"
        judge_t3_records(
            generation_records,
            {str(row["example_id"]): row for row in rows},
            judge_config,
            cache_path,
            lambda value: extract_tag(value, "misconception"),
            normalize_text,
        )
        write_audit_sample(
            args.output_dir / "t3_judge_audit_sample.jsonl",
            generation_records,
            args.judge_audit_samples,
            args.seed,
        )
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "eval_jsonl": str(args.eval_jsonl),
        "artifact_sha256": {
            "adapter_model": (
                artifact_sha256(Path(args.adapter) / "adapter_model.safetensors")
                if args.adapter is not None
                else None
            ),
            "evaluation_input": artifact_sha256(args.eval_jsonl),
        },
        "max_samples_per_group": args.max_samples_per_group,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "generation_batch_size": args.generation_batch_size,
        "sampled_rows": len(rows),
        "unsharded_filtered_rows": unsharded_rows,
        "evaluation_shard": (
            None
            if args.shard_index is None
            else {"index": args.shard_index, "count": args.num_shards}
        ),
        "classification_rows": len(classification_rows),
        "generation_rows": len(generation_rows),
        "bottleneck_mask": args.bottleneck_mask,
        "bottleneck_style": args.bottleneck_style if args.bottleneck_mask else None,
        "metrics_by_group": summarize(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "predictions_redacted.jsonl", redacted_records(records))
    if args.save_unredacted_predictions:
        write_jsonl(args.output_dir / "predictions.jsonl", records)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
