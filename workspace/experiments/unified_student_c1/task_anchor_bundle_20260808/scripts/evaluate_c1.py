#!/usr/bin/env python3
"""Evaluate C1 unified-generation samples with baselines and optional CausalLM."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

STATE_TOKEN = "<STUDENT_STATE>"
HISTORY_OPEN = "<student_history>"
BOTTLENECK_STYLES = ("standard", "late_fusion")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def key(row: dict[str, Any], name: str) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get(name) or "")


def labels_for_row(row: dict[str, Any]) -> list[str]:
    return list(row["candidate_labels"])


def labels_for_rows(rows: list[dict[str, Any]], preds: list[str] | None = None) -> list[str]:
    seen: set[str] = set()
    labels: list[str] = []
    for row in rows:
        for label in labels_for_row(row):
            if label not in seen:
                seen.add(label)
                labels.append(label)
    if preds is not None:
        for label in preds:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def candidate_text(row: dict[str, Any], label: str) -> str:
    return str(row["candidate_targets"][label])


def encode_prompt_text(prompt: str, state_token_alias: str | None) -> str:
    return prompt.replace(STATE_TOKEN, state_token_alias) if state_token_alias else prompt


def locate_bottleneck_positions(
    tokenizer, prompt: str, state_token_alias: str | None = None
) -> tuple[int, int]:
    state_char = prompt.find(STATE_TOKEN)
    history_char = prompt.find(HISTORY_OPEN)
    if state_char < 0:
        raise ValueError(f"Could not find {STATE_TOKEN} in prompt")
    if history_char < 0:
        raise ValueError(f"Could not find {HISTORY_OPEN} in prompt")
    history_start = len(tokenizer(prompt[:history_char], add_special_tokens=False)["input_ids"])
    through_state = prompt[:state_char] + (state_token_alias or STATE_TOKEN)
    state_pos = len(tokenizer(through_state, add_special_tokens=False)["input_ids"]) - 1
    if history_start >= state_pos:
        raise ValueError(f"Invalid bottleneck positions: history_start={history_start}, state_pos={state_pos}")
    return history_start, state_pos


def build_bottleneck_attention_mask(
    attention_2d: torch.Tensor,
    history_starts: torch.Tensor,
    state_positions: torch.Tensor,
    prompt_ends: torch.Tensor | None = None,
    style: str = "standard",
) -> torch.Tensor:
    if style not in BOTTLENECK_STYLES:
        raise ValueError(f"Unknown bottleneck style: {style}")
    batch_size, seq_len = attention_2d.shape
    device = attention_2d.device
    min_value = torch.finfo(torch.float32).min
    causal_block = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    mask.masked_fill_(causal_block.view(1, 1, seq_len, seq_len), min_value)
    mask.masked_fill_(attention_2d.eq(0).view(batch_size, 1, 1, seq_len), min_value)
    for b in range(batch_size):
        h0 = int(history_starts[b].item())
        state_pos = int(state_positions[b].item())
        valid_len = int(attention_2d[b].sum().item())
        if not (0 <= h0 < state_pos < valid_len):
            raise ValueError(f"Invalid mask positions: h0={h0}, state_pos={state_pos}, valid_len={valid_len}")
        mask[b, 0, state_pos + 1 : valid_len, h0:state_pos] = min_value
        if style == "late_fusion":
            if prompt_ends is None:
                prompt_end = valid_len
            else:
                prompt_end = int(prompt_ends[b].item())
            if not (state_pos < prompt_end <= valid_len):
                raise ValueError(
                    f"Invalid prompt_end for late_fusion: state_pos={state_pos}, "
                    f"prompt_end={prompt_end}, valid_len={valid_len}"
                )
            mask[b, 0, state_pos + 1 : prompt_end, state_pos] = min_value
    return mask


def target_label(row: dict[str, Any]) -> str:
    return str(row["target_label"])


def macro_f1(golds: list[str], preds: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / max(1, len(scores))


def binary_auc(golds: list[str], probs: list[dict[str, float]], positive: str) -> float | None:
    pairs = [(float(prob.get(positive, 0.0)), 1 if gold == positive else 0) for gold, prob in zip(golds, probs)]
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(golds: list[str], probs: list[dict[str, float]], bins: int = 10) -> float:
    buckets = [[] for _ in range(bins)]
    for gold, prob in zip(golds, probs):
        pred, conf = max(prob.items(), key=lambda kv: kv[1])
        idx = min(bins - 1, int(conf * bins))
        buckets[idx].append((1.0 if pred == gold else 0.0, conf))
    total = len(golds)
    value = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        acc = sum(x for x, _ in bucket) / len(bucket)
        conf = sum(c for _, c in bucket) / len(bucket)
        value += len(bucket) / max(1, total) * abs(acc - conf)
    return value


def summarize(rows: list[dict[str, Any]], preds: list[str], probs: list[dict[str, float]] | None = None) -> dict[str, Any]:
    golds = [target_label(row) for row in rows]
    labels = labels_for_rows(rows, preds) if rows else []
    out: dict[str, Any] = {
        "n": len(rows),
        "accuracy": sum(g == p for g, p in zip(golds, preds)) / max(1, len(rows)),
        "macro_f1": macro_f1(golds, preds, labels),
        "target_counts": dict(Counter(golds)),
        "prediction_counts": dict(Counter(preds)),
    }
    if probs is not None:
        nll = []
        brier = []
        for gold, prob in zip(golds, probs):
            nll.append(-math.log(max(float(prob.get(gold, 0.0)), 1e-12)))
            row_labels = list(prob)
            if len(row_labels) == 2:
                pos = row_labels[0]
                y = 1.0 if gold == pos else 0.0
                brier.append((float(prob.get(pos, 0.0)) - y) ** 2)
        out["nll"] = sum(nll) / max(1, len(nll))
        out["ece"] = ece(golds, probs)
        if brier:
            out["brier"] = sum(brier) / len(brier)
            binary_labels = labels_for_row(rows[0])
            if len(binary_labels) == 2 and all(labels_for_row(row) == binary_labels for row in rows):
                out["auc_positive_" + binary_labels[0]] = binary_auc(golds, probs, binary_labels[0])
    return out


def make_count_tables(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tables: dict[str, Any] = {
        "global": defaultdict(Counter),
        "skill": defaultdict(lambda: defaultdict(Counter)),
        "question": defaultdict(lambda: defaultdict(Counter)),
    }
    for row in train_rows:
        task = row["task"]
        label = target_label(row)
        tables["global"][task][label] += 1
        tables["skill"][task][key(row, "skill_id")][label] += 1
        tables["question"][task][key(row, "question_id")][label] += 1
    return tables


def probs_from_counts(counts: Counter[str], labels: list[str], smoothing: float = 1.0) -> dict[str, float]:
    denom = sum(counts.values()) + smoothing * len(labels)
    return {label: (counts.get(label, 0) + smoothing) / denom for label in labels}


def history_prediction(row: dict[str, Any]) -> tuple[str, dict[str, float]]:
    labels = labels_for_row(row)
    task = row["task"]
    skill = key(row, "skill_id")
    history = row.get("history") or []
    same_skill = [h for h in history if str(h.get("skill_id") or h.get("family") or "") == skill]
    pool = same_skill or history
    if task == "next_correctness" or task.endswith("next_correctness"):
        n = len(pool)
        correct = sum(int(h.get("correctness", h.get("is_correct", 0))) for h in pool)
        p_correct = (correct + 1) / (n + 2)
        probs = {"correct": p_correct, "incorrect": 1 - p_correct}
        return max(probs, key=probs.get), probs
    if task == "next_response" or task.endswith("next_response"):
        counts = Counter(str(h.get("student_response")) for h in pool if h.get("student_response"))
        probs = probs_from_counts(counts, labels)
        return max(probs, key=probs.get), probs
    if task == "future_mastery":
        n = len(pool)
        correct = sum(int(h.get("correctness", h.get("is_correct", 0))) for h in pool)
        rate = (correct + 1) / (n + 2)
        pred = "low" if rate < 0.4 else "medium" if rate < 0.7 else "high"
        probs = {label: 0.1 for label in labels}
        probs[pred] = 0.8
        z = sum(probs.values())
        return pred, {k: v / z for k, v in probs.items()}
    probs = {label: 1.0 / len(labels) for label in labels}
    return labels[0], probs


def baseline_metrics(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tables = make_count_tables(train_rows)
    out: dict[str, Any] = {}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        by_task[row["task"]].append(row)

    for task, rows in sorted(by_task.items()):
        task_out: dict[str, Any] = {}
        uniform_probs = []
        uniform_preds = []
        uniform_acc = []
        for row in rows:
            labels = labels_for_row(row)
            uniform_probs.append({label: 1.0 / len(labels) for label in labels})
            uniform_preds.append(labels[0])
            uniform_acc.append(1.0 / len(labels))
        task_out["random_uniform"] = summarize(rows, uniform_preds, uniform_probs) | {
            "accuracy": sum(uniform_acc) / max(1, len(uniform_acc)),
            "note": "Expected accuracy under uniform random guessing.",
        }
        global_counts = tables["global"][task]
        global_probs = [probs_from_counts(global_counts, labels_for_row(row)) for row in rows]
        task_out["global_majority"] = summarize(rows, [max(p, key=p.get) for p in global_probs], global_probs)

        skill_probs = []
        for row in rows:
            counts = tables["skill"][task].get(key(row, "skill_id"), global_counts)
            skill_probs.append(probs_from_counts(counts, labels_for_row(row)))
        task_out["skill_majority"] = summarize(rows, [max(p, key=p.get) for p in skill_probs], skill_probs)

        question_probs = []
        for row in rows:
            counts = tables["question"][task].get(key(row, "question_id"))
            if counts is None:
                counts = tables["skill"][task].get(key(row, "skill_id"), global_counts)
            question_probs.append(probs_from_counts(counts, labels_for_row(row)))
        task_out["question_majority"] = summarize(rows, [max(p, key=p.get) for p in question_probs], question_probs)

        hist_preds, hist_probs = [], []
        for row in rows:
            pred, prob = history_prediction(row)
            hist_preds.append(pred)
            hist_probs.append(prob)
        task_out["history_state"] = summarize(rows, hist_preds, hist_probs)
        task_out["n_skill_entries"] = len(tables["skill"][task])
        task_out["n_question_entries"] = len(tables["question"][task])
        out[task] = task_out
    return out


def tokenizer_source(model_path: str, adapter_path: str | None) -> str:
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if (adapter_dir / "tokenizer.json").exists() or (adapter_dir / "tokenizer_config.json").exists():
            return str(adapter_dir)
    return model_path


def load_model(
    model_path: str,
    adapter_path: str | None,
    bottleneck_mask: bool = False,
    load_4bit: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source(model_path, adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if bottleneck_mask:
        model_kwargs["attn_implementation"] = "eager"
    if load_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        **model_kwargs,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def encode_candidate(
    tokenizer,
    row: dict[str, Any],
    label: str,
    max_length: int,
    bottleneck_mask: bool = False,
    state_token_alias: str | None = None,
) -> dict[str, Any]:
    raw_prefix = str(row["prompt"])
    prefix = encode_prompt_text(raw_prefix, state_token_alias)
    target = candidate_text(row, label)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    history_start, state_pos = (-1, -1)
    if bottleneck_mask:
        history_start, state_pos = locate_bottleneck_positions(tokenizer, raw_prefix, state_token_alias)
    if len(prefix_ids) + len(target_ids) > max_length:
        trim = len(prefix_ids) + len(target_ids) - max_length
        keep = max_length - len(target_ids)
        prefix_ids = prefix_ids[-keep:]
        if bottleneck_mask:
            history_start -= trim
            state_pos -= trim
            if history_start < 0 or state_pos < 0:
                raise ValueError(f"Trimming removed bottleneck positions for {row.get('example_id')}")
    return {
        "input_ids": prefix_ids + target_ids,
        "prefix_len": len(prefix_ids),
        "target_len": len(target_ids),
        "label": label,
        "history_start": history_start,
        "state_pos": state_pos,
    }


@torch.inference_mode()
def model_metrics(
    model_path: str,
    adapter_path: str | None,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    progress_every: int,
    bottleneck_mask: bool = False,
    bottleneck_style: str = "standard",
    state_token_alias: str | None = None,
    load_4bit: bool = False,
) -> dict[str, Any]:
    tokenizer, model = load_model(model_path, adapter_path, bottleneck_mask, load_4bit)
    device = getattr(model, "device", next(model.parameters()).device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    out: dict[str, Any] = {}
    for task, task_rows in sorted(by_task.items()):
        all_items = []
        for row_idx, row in enumerate(task_rows):
            for label in labels_for_row(row):
                item = encode_candidate(
                    tokenizer, row, label, max_length, bottleneck_mask, state_token_alias
                )
                item["row_idx"] = row_idx
                all_items.append(item)

        scores: list[dict[str, float]] = [dict() for _ in task_rows]
        total_batches = (len(all_items) + batch_size - 1) // batch_size
        print(
            f"[model_eval] task={task} rows={len(task_rows)} candidates={len(all_items)} batches={total_batches}",
            flush=True,
        )
        for start in range(0, len(all_items), batch_size):
            batch = all_items[start : start + batch_size]
            batch_idx = start // batch_size + 1
            if progress_every > 0 and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches):
                print(f"[model_eval] task={task} batch={batch_idx}/{total_batches}", flush=True)
            max_len = max(len(item["input_ids"]) for item in batch)
            input_ids, attn, history_starts, state_positions, prompt_ends = [], [], [], [], []
            for item in batch:
                ids = item["input_ids"]
                pad = max_len - len(ids)
                input_ids.append(ids + [pad_id] * pad)
                attn.append([1] * len(ids) + [0] * pad)
                history_starts.append(int(item["history_start"]))
                state_positions.append(int(item["state_pos"]))
                prompt_ends.append(int(item["prefix_len"]))
            input_tensor = torch.tensor(input_ids, device=device)
            attn_tensor_2d = torch.tensor(attn, device=device)
            if bottleneck_mask:
                attn_tensor = build_bottleneck_attention_mask(
                    attn_tensor_2d,
                    torch.tensor(history_starts, dtype=torch.long, device=device),
                    torch.tensor(state_positions, dtype=torch.long, device=device),
                    torch.tensor(prompt_ends, dtype=torch.long, device=device),
                    bottleneck_style,
                )
            else:
                attn_tensor = attn_tensor_2d
            logits = model(input_ids=input_tensor, attention_mask=attn_tensor).logits
            for i, item in enumerate(batch):
                ids = item["input_ids"]
                prefix_len = item["prefix_len"]
                score = 0.0
                for pos in range(prefix_len, len(ids)):
                    logp = torch.log_softmax(logits[i, pos - 1, :].float(), dim=-1)
                    score += float(logp[ids[pos]].item())
                scores[item["row_idx"]][item["label"]] = score

        probs, preds = [], []
        for row, score in zip(task_rows, scores):
            labels = labels_for_row(row)
            values = torch.tensor([score[label] for label in labels], dtype=torch.float32)
            p = torch.softmax(values, dim=0)
            prob = {label: float(p[i].item()) for i, label in enumerate(labels)}
            probs.append(prob)
            preds.append(max(prob, key=prob.get))
        by_source: dict[str, Any] = {}
        source_indices: dict[str, list[int]] = defaultdict(list)
        for row_index, row in enumerate(task_rows):
            source_indices[str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")].append(row_index)
        for source_name, indices in sorted(source_indices.items()):
            by_source[source_name] = summarize(
                [task_rows[index] for index in indices],
                [preds[index] for index in indices],
                [probs[index] for index in indices],
            )
        out[task] = summarize(task_rows, preds, probs) | {
            "scoring_mode": "candidate_target_logprob",
            "bottleneck_mask": bottleneck_mask,
            "bottleneck_style": bottleneck_style if bottleneck_mask else None,
            "load_4bit": load_4bit,
            "by_source": by_source,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--max-samples-per-task", type=int)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--bottleneck-mask", action="store_true")
    parser.add_argument("--bottleneck-style", choices=BOTTLENECK_STYLES, default="standard")
    parser.add_argument("--state-token-alias")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_jsonl)
    eval_all = read_jsonl(args.eval_jsonl)
    if args.max_samples_per_task:
        kept = []
        counts = Counter()
        for row in eval_all:
            if counts[row["task"]] < args.max_samples_per_task:
                kept.append(row)
                counts[row["task"]] += 1
        eval_rows = kept
    else:
        eval_rows = eval_all
    if args.tasks:
        task_set = set(args.tasks)
        eval_rows = [row for row in eval_rows if row["task"] in task_set]
        if not eval_rows:
            raise SystemExit(f"No eval rows left after --tasks filter: {sorted(task_set)}")

    result: dict[str, Any] = {
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "max_samples_per_task": args.max_samples_per_task,
        "baselines": baseline_metrics(train_rows, eval_rows),
    }
    if not args.skip_model:
        if not args.model:
            raise SystemExit("--model is required unless --skip-model is set")
        result["model"] = {
            "model_path": args.model,
            "adapter_path": args.adapter,
            "metrics": model_metrics(
                args.model,
                args.adapter,
                eval_rows,
                args.max_length,
                args.batch_size,
                args.progress_every,
                args.bottleneck_mask,
                args.bottleneck_style,
                args.state_token_alias,
                args.load_4bit,
            ),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    write_json(args.output_json, result)


if __name__ == "__main__":
    main()
