#!/usr/bin/env python3
"""Linear-probe smoke test for <STUDENT_STATE> activations on Malrule data."""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


STATE_TOKEN = "<STUDENT_STATE>"
HISTORY_OPEN = "<student_history>"
BOTTLENECK_STYLES = ("standard", "late_fusion")


def get_decoder_layers(model):
    """Return the decoder block list for bare HF models or PEFT-wrapped models."""
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("base_model.model.model.layers", lambda m: m.base_model.model.model.layers),
    ]
    for _name, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise AttributeError("Could not find decoder layers on model")


def tokenizer_source(model_path: str, adapter_path: str | None) -> str:
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if (adapter_dir / "tokenizer.json").exists() or (adapter_dir / "tokenizer_config.json").exists():
            return str(adapter_dir)
    return model_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def label_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("malrule_id") or "NONE")


def family_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("family") or "NONE")


def stratified_sample(rows: list[dict[str, Any]], per_label: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[label_of(row)].append(row)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for label in sorted(buckets):
        items = buckets[label]
        rng.shuffle(items)
        out.extend(items[:per_label])
    rng.shuffle(out)
    return out


def filter_task(rows: list[dict[str, Any]], task: str | None) -> list[dict[str, Any]]:
    if not task:
        return rows
    return [row for row in rows if row.get("task") == task]


def find_subsequence(xs: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    for i in range(0, len(xs) - len(needle) + 1):
        if xs[i : i + len(needle)] == needle:
            return i
    return -1


def encode_prompt_text(prompt: str, state_token_alias: str | None) -> str:
    if not state_token_alias:
        return prompt
    if STATE_TOKEN not in prompt:
        raise ValueError(f"Could not locate {STATE_TOKEN} in prompt text")
    return prompt.replace(STATE_TOKEN, state_token_alias)


def locate_positions(
    tokenizer,
    prompt: str,
    state_token_alias: str | None = None,
) -> tuple[list[int], int, int, int]:
    marker_char = prompt.find(STATE_TOKEN)
    if marker_char < 0:
        raise ValueError(f"Could not locate {STATE_TOKEN} in prompt text")
    history_char = prompt.find(HISTORY_OPEN)
    if history_char < 0:
        raise ValueError(f"Could not locate {HISTORY_OPEN} in prompt text")
    marker_text = state_token_alias or STATE_TOKEN
    encoded_prompt = encode_prompt_text(prompt, state_token_alias)
    prefix_ids = tokenizer(prompt[:marker_char], add_special_tokens=False)["input_ids"]
    through_marker_ids = tokenizer(prompt[:marker_char] + marker_text, add_special_tokens=False)["input_ids"]
    ids = tokenizer(encoded_prompt, add_special_tokens=False)["input_ids"]
    state_position = len(through_marker_ids) - 1
    history_position = max(0, len(prefix_ids) - 1)
    history_start = len(tokenizer(prompt[:history_char], add_special_tokens=False)["input_ids"])
    if history_start >= state_position:
        raise ValueError(f"Invalid positions: history_start={history_start}, state_position={state_position}")
    return ids, state_position, history_position, history_start


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


def trim_around_state(
    ids: list[int], state_position: int, history_position: int, max_length: int
) -> tuple[list[int], int, int]:
    if len(ids) <= max_length:
        return ids, state_position, history_position
    # Keep the marker and as much surrounding context as possible. In the
    # current Malrule smoke data this branch should rarely be used, but it
    # prevents long future prompts from silently dropping the marker.
    desired_left = max_length // 2
    start = max(0, min(state_position - desired_left, len(ids) - max_length))
    end = start + max_length
    if not (start <= state_position < end):
        raise ValueError(f"Could not preserve {STATE_TOKEN} while trimming")
    trimmed = ids[start:end]
    return trimmed, state_position - start, max(0, history_position - start)


def encode_batch(
    tokenizer,
    rows: list[dict[str, Any]],
    max_length: int,
    state_token_alias: str | None = None,
):
    encoded = []
    state_positions = []
    history_positions = []
    history_starts = []
    prompt_ends = []
    for row in rows:
        ids, state_pos, hist_pos, history_start = locate_positions(
            tokenizer,
            str(row["prompt"]),
            state_token_alias,
        )
        ids, state_pos, hist_pos = trim_around_state(ids, state_pos, hist_pos, max_length)
        if history_start > 0 and len(ids) <= max_length:
            trimmed_history_start = history_start
        else:
            trimmed_history_start = max(0, min(history_start, state_pos - 1))
        state_positions.append(state_pos)
        history_positions.append(hist_pos)
        history_starts.append(trimmed_history_start)
        prompt_ends.append(len(ids))
        encoded.append(ids)
    max_len = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    attention_mask = []
    for ids in encoded:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(state_positions, dtype=torch.long),
        torch.tensor(history_positions, dtype=torch.long),
        torch.tensor(history_starts, dtype=torch.long),
        torch.tensor(prompt_ends, dtype=torch.long),
    )


@torch.inference_mode()
def collect_activations(
    model_path: str,
    adapter_path: str | None,
    rows: list[dict[str, Any]],
    layers: list[int],
    max_length: int,
    batch_size: int,
    bottleneck_mask: bool = False,
    bottleneck_style: str = "standard",
    state_token_alias: str | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source(model_path, adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    marker_text = state_token_alias or STATE_TOKEN
    marker_ids = tokenizer(marker_text, add_special_tokens=False)["input_ids"]
    if state_token_alias and len(marker_ids) != 1:
        raise ValueError(f"--state-token-alias must tokenize to one id, got {marker_ids}")
    print(
        f"[probe] marker={marker_text!r} marker_ids={marker_ids} "
        f"pieces={[tokenizer.decode([i]) for i in marker_ids]}",
        flush=True,
    )

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = {}
    if bottleneck_mask:
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        **model_kwargs,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print(
            f"[probe] resize model embeddings {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}",
            flush=True,
        )
        model.resize_token_embeddings(len(tokenizer))
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    device = getattr(model, "device", next(model.parameters()).device)

    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range for model with {n_layers} layers")
    print(f"[probe] model_layers={n_layers} selected_layers={layers}", flush=True)

    current_state_positions: torch.Tensor | None = None
    current_history_positions: torch.Tensor | None = None
    batch_capture_state: dict[int, torch.Tensor] = {}
    batch_capture_history: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            nonlocal current_state_positions, current_history_positions
            hidden = output[0] if isinstance(output, tuple) else output
            assert current_state_positions is not None
            assert current_history_positions is not None
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            state_pos = current_state_positions.to(hidden.device)
            hist_pos = current_history_positions.to(hidden.device)
            batch_capture_state[layer_idx] = hidden[batch_idx, state_pos].detach().float().cpu()
            batch_capture_history[layer_idx] = hidden[batch_idx, hist_pos].detach().float().cpu()

        return hook

    handles = [decoder_layers[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    state_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    hist_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    try:
        total_batches = (len(rows) + batch_size - 1) // batch_size
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_idx = start // batch_size + 1
            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
                print(f"[probe] activation batch {batch_idx}/{total_batches}", flush=True)
            input_ids, attention_mask, state_pos, hist_pos, history_start, prompt_end = encode_batch(
                tokenizer,
                batch_rows,
                max_length,
                state_token_alias,
            )
            current_state_positions = state_pos
            current_history_positions = hist_pos
            batch_capture_state = {}
            batch_capture_history = {}
            attention_on_device = attention_mask.to(device)
            if bottleneck_mask:
                attention_for_model = build_bottleneck_attention_mask(
                    attention_on_device,
                    history_start.to(device),
                    state_pos.to(device),
                    prompt_end.to(device),
                    bottleneck_style,
                )
            else:
                attention_for_model = attention_on_device
            _ = model(input_ids=input_ids.to(device), attention_mask=attention_for_model, use_cache=False)
            for layer in layers:
                state_chunks[layer].append(batch_capture_state[layer])
                hist_chunks[layer].append(batch_capture_history[layer])
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "state": {layer: torch.cat(state_chunks[layer]).numpy() for layer in layers},
        "history_last": {layer: torch.cat(hist_chunks[layer]).numpy() for layer in layers},
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / max(1, len(scores)))


def train_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(y_train.max()) + 1
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    xtr = torch.tensor((x_train - mean) / std, dtype=torch.float32, device=device)
    xev = torch.tensor((x_eval - mean) / std, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    yev = torch.tensor(y_eval, dtype=torch.long, device=device)
    clf = torch.nn.Linear(xtr.shape[1], n_classes, device=device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(clf(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = clf(xev).argmax(dim=-1).cpu().numpy()
    y_eval_np = yev.cpu().numpy()
    return {
        "fit_device": str(device),
        "accuracy": float((pred == y_eval_np).mean()),
        "macro_f1": macro_f1(y_eval_np, pred, n_classes),
        "prediction_counts": dict(Counter(map(int, pred))),
    }


def encode_labels(train_rows: list[dict[str, Any]], *eval_groups: list[dict[str, Any]]):
    labels = sorted({label_of(row) for row in train_rows})
    label_to_id = {label: i for i, label in enumerate(labels)}
    encoded = []
    for rows in (train_rows, *eval_groups):
        encoded.append(np.array([label_to_id[label_of(row)] for row in rows], dtype=np.int64))
    return labels, encoded


def family_breakdown(rows: list[dict[str, Any]], y_true: np.ndarray, y_pred_counts: dict[str, Any] | None = None):
    del y_pred_counts
    families = sorted({family_of(row) for row in rows})
    return {family: sum(family_of(row) == family for row in rows) for family in families}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--same-jsonl", type=Path, required=True)
    parser.add_argument("--cross-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--task", help="Optional task-name filter before sampling.")
    parser.add_argument("--layers", default="0,8,16,24,32,40,48,56,63")
    parser.add_argument("--train-per-label", type=int, default=50)
    parser.add_argument("--eval-per-label", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bottleneck-mask", action="store_true")
    parser.add_argument("--bottleneck-style", choices=BOTTLENECK_STYLES, default="standard")
    parser.add_argument(
        "--state-token-alias",
        help=f"Token used in place of {STATE_TOKEN} during training, e.g. <|vision_pad|>.",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    train_pool = filter_task(read_jsonl(args.train_jsonl), args.task)
    same_pool = filter_task(read_jsonl(args.same_jsonl), args.task)
    cross_pool = filter_task(read_jsonl(args.cross_jsonl), args.task)
    train_rows = stratified_sample(train_pool, args.train_per_label, args.seed)
    same_rows = stratified_sample(same_pool, args.eval_per_label, args.seed + 1)
    cross_rows = stratified_sample(cross_pool, args.eval_per_label, args.seed + 2)
    labels, encoded = encode_labels(train_rows, same_rows, cross_rows)
    y_train, y_same, y_cross = encoded
    print(
        f"[probe] rows train={len(train_rows)} same={len(same_rows)} cross={len(cross_rows)} labels={len(labels)}",
        flush=True,
    )

    train_acts = collect_activations(
        args.model,
        args.adapter,
        train_rows,
        layers,
        args.max_length,
        args.batch_size,
        args.bottleneck_mask,
        args.bottleneck_style,
        args.state_token_alias,
    )
    same_acts = collect_activations(
        args.model,
        args.adapter,
        same_rows,
        layers,
        args.max_length,
        args.batch_size,
        args.bottleneck_mask,
        args.bottleneck_style,
        args.state_token_alias,
    )
    cross_acts = collect_activations(
        args.model,
        args.adapter,
        cross_rows,
        layers,
        args.max_length,
        args.batch_size,
        args.bottleneck_mask,
        args.bottleneck_style,
        args.state_token_alias,
    )

    results: dict[str, Any] = {
        "model": args.model,
        "adapter": args.adapter,
        "train_jsonl": str(args.train_jsonl),
        "same_jsonl": str(args.same_jsonl),
        "cross_jsonl": str(args.cross_jsonl),
        "task": args.task,
        "layers": layers,
        "bottleneck_mask": args.bottleneck_mask,
        "bottleneck_style": args.bottleneck_style if args.bottleneck_mask else None,
        "state_token_alias": args.state_token_alias,
        "n_labels": len(labels),
        "labels": labels,
        "label_counts": {
            "train": dict(Counter(label_of(r) for r in train_rows)),
            "same": dict(Counter(label_of(r) for r in same_rows)),
            "cross": dict(Counter(label_of(r) for r in cross_rows)),
        },
        "family_counts": {
            "train": dict(Counter(family_of(r) for r in train_rows)),
            "same": dict(Counter(family_of(r) for r in same_rows)),
            "cross": dict(Counter(family_of(r) for r in cross_rows)),
        },
        "random_level": 1.0 / len(labels),
        "metrics": {},
    }

    for position_name in ["state", "history_last"]:
        results["metrics"][position_name] = {}
        for layer in layers:
            print(f"[probe] train probe position={position_name} layer={layer}", flush=True)
            same = train_linear_probe(
                train_acts[position_name][layer],
                y_train,
                same_acts[position_name][layer],
                y_same,
                args.seed,
                args.epochs,
                args.lr,
                args.weight_decay,
            )
            cross = train_linear_probe(
                train_acts[position_name][layer],
                y_train,
                cross_acts[position_name][layer],
                y_cross,
                args.seed,
                args.epochs,
                args.lr,
                args.weight_decay,
            )
            results["metrics"][position_name][str(layer)] = {"same": same, "cross": cross}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results["metrics"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
