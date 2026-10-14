#!/usr/bin/env python3
"""Activation-swap smoke test for Malrule misconception diagnosis."""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


STATE_TOKEN = "<STUDENT_STATE>"
HISTORY_OPEN = "<student_history>"
BOTTLENECK_STYLES = ("standard", "late_fusion")


def read_jsonl(path: Path, task: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") == task:
                rows.append(row)
    return rows


def label_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("malrule_id") or "NONE")


def family_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("family") or "NONE")


def tokenizer_source(model_path: str, adapter_path: str | None) -> str:
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if (adapter_dir / "tokenizer.json").exists() or (adapter_dir / "tokenizer_config.json").exists():
            return str(adapter_dir)
    return model_path


def get_decoder_layers(model):
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.base_model.model.model.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise AttributeError("Could not find decoder layers on model")


def encode_prompt_text(prompt: str, state_token_alias: str | None) -> str:
    if not state_token_alias:
        return prompt
    if STATE_TOKEN not in prompt:
        raise ValueError(f"Could not locate {STATE_TOKEN}")
    return prompt.replace(STATE_TOKEN, state_token_alias)


def locate_state_position(
    tokenizer,
    prompt: str,
    state_token_alias: str | None = None,
) -> tuple[list[int], int]:
    marker_char = prompt.find(STATE_TOKEN)
    if marker_char < 0:
        raise ValueError(f"Could not locate {STATE_TOKEN}")
    marker_text = state_token_alias or STATE_TOKEN
    through_marker_ids = tokenizer(prompt[:marker_char] + marker_text, add_special_tokens=False)["input_ids"]
    ids = tokenizer(encode_prompt_text(prompt, state_token_alias), add_special_tokens=False)["input_ids"]
    state_pos = len(through_marker_ids) - 1
    state_ids = tokenizer(marker_text, add_special_tokens=False)["input_ids"]
    if state_token_alias and len(state_ids) != 1:
        raise ValueError(f"--state-token-alias must tokenize to one id, got {state_ids}")
    if len(state_ids) == 1 and ids[state_pos] != state_ids[0]:
        raise ValueError(f"State position mismatch: got token {ids[state_pos]}, expected {state_ids[0]}")
    return ids, state_pos


def locate_bottleneck_positions(
    tokenizer,
    prompt: str,
    state_token_alias: str | None = None,
) -> tuple[list[int], int, int]:
    ids, state_pos = locate_state_position(tokenizer, prompt, state_token_alias)
    history_char = prompt.find(HISTORY_OPEN)
    if history_char < 0:
        raise ValueError(f"Could not locate {HISTORY_OPEN}")
    history_start = len(tokenizer(prompt[:history_char], add_special_tokens=False)["input_ids"])
    if history_start >= state_pos:
        raise ValueError(f"Invalid bottleneck positions: history_start={history_start}, state_pos={state_pos}")
    return ids, history_start, state_pos


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


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    out = [int(x) for x in spec.split(",") if x.strip()]
    for layer in out:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range 0..{n_layers - 1}")
    return out


def parse_layer_groups(spec: str, n_layers: int) -> list[tuple[str, list[int]]]:
    """Parse named patch groups such as ``late=48-63;final=63``."""
    groups: list[tuple[str, list[int]]] = []
    seen_names: set[str] = set()
    for raw_group in spec.split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        if "=" not in raw_group:
            raise ValueError(f"Layer group lacks name=spec: {raw_group!r}")
        name, raw_layers = (part.strip() for part in raw_group.split("=", 1))
        if not name or name in seen_names:
            raise ValueError(f"Layer group name is empty or duplicated: {name!r}")
        values: list[int] = []
        for item in raw_layers.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"Descending layer range: {item!r}")
                values.extend(range(start, end + 1))
            else:
                values.append(int(item))
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Layer group must be non-empty and unique: {raw_group!r}")
        if any(layer < 0 or layer >= n_layers for layer in values):
            raise ValueError(f"Layer group {name!r} outside model depth={n_layers}")
        seen_names.add(name)
        groups.append((name, values))
    if not groups:
        raise ValueError("--layer-groups produced no groups")
    return groups


def response_answer_for_malrule(row: dict[str, Any], malrule_id: str) -> str | None:
    meta = row.get("metadata") or {}
    answers = meta.get("query_answers_by_malrule") or {}
    value = answers.get(malrule_id)
    return None if value is None else str(value)


def pair_targets(
    donor: dict[str, Any],
    recipient: dict[str, Any],
    swap_target: str,
) -> tuple[str, str, str | None]:
    donor_label = label_of(donor)
    recipient_label = label_of(recipient)
    if swap_target == "identity":
        return donor_label, recipient_label, None
    if swap_target == "response":
        donor_answer = response_answer_for_malrule(recipient, donor_label)
        recipient_answer = str(recipient.get("target_label"))
        correct = recipient.get("query", {}).get("correct_answer")
        return str(donor_answer) if donor_answer is not None else "", recipient_answer, (
            str(correct) if correct is not None else None
        )
    raise ValueError(f"Unsupported swap_target={swap_target}")


def valid_pair_for_target(donor: dict[str, Any], recipient: dict[str, Any], swap_target: str) -> bool:
    donor_label = label_of(donor)
    recipient_label = label_of(recipient)
    if donor_label == recipient_label:
        return False
    if family_of(donor) != family_of(recipient):
        return False
    if swap_target == "identity":
        return True
    donor_answer, recipient_answer, _correct = pair_targets(donor, recipient, swap_target)
    candidate_labels = {str(x) for x in recipient.get("candidate_labels", [])}
    if not donor_answer or donor_answer not in candidate_labels or recipient_answer not in candidate_labels:
        return False
    correct = recipient.get("query", {}).get("correct_answer")
    correct_answer = str(correct) if correct is not None else None
    if correct_answer is not None and (donor_answer == correct_answer or recipient_answer == correct_answer):
        return False
    return donor_answer != recipient_answer


def build_pairs(
    rows: list[dict[str, Any]],
    num_pairs: int,
    seed: int,
    swap_target: str = "identity",
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rng = random.Random(seed)
    candidates = [row for row in rows if label_of(row) != "NONE"]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for recipient in candidates:
        for donor in candidates:
            if donor["example_id"] == recipient["example_id"]:
                continue
            if valid_pair_for_target(donor, recipient, swap_target):
                pairs.append((donor, recipient))
    rng.shuffle(pairs)
    pairs = pairs[:num_pairs]
    if len(pairs) < num_pairs:
        print(f"[swap] warning: requested {num_pairs} pairs, built {len(pairs)}", flush=True)
    return pairs


def build_samemal_controls(
    rows: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    seed: int,
) -> list[dict[str, Any] | None]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = label_of(row)
        if label == "NONE":
            continue
        buckets[(family_of(row), label)].append(row)

    controls: list[dict[str, Any] | None] = []
    missing = 0
    for _donor, recipient in pairs:
        candidates = [
            row
            for row in buckets[(family_of(recipient), label_of(recipient))]
            if row["example_id"] != recipient["example_id"]
        ]
        if not candidates:
            controls.append(None)
            missing += 1
        else:
            controls.append(rng.choice(candidates))
    if missing:
        print(f"[swap] warning: samemal missing for {missing}/{len(pairs)} pairs", flush=True)
    return controls


def build_real_random_controls(
    rows: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    samemal_controls: list[dict[str, Any] | None],
    seed: int,
) -> list[dict[str, Any]]:
    """Select an in-distribution donor without using model outputs.

    The control is matched on MalRule family but has a rule identity different
    from both the directed donor and recipient.  Sorting before seeded sampling
    makes the selection invariant to JSONL row order.
    """
    rng = random.Random(seed)
    controls: list[dict[str, Any]] = []
    for (donor, recipient), samemal in zip(pairs, samemal_controls, strict=True):
        excluded_ids = {str(donor["example_id"]), str(recipient["example_id"])}
        if samemal is not None:
            excluded_ids.add(str(samemal["example_id"]))
        excluded_labels = {label_of(donor), label_of(recipient)}
        candidates = sorted(
            (
                row
                for row in rows
                if family_of(row) == family_of(recipient)
                and label_of(row) not in excluded_labels
                and str(row["example_id"]) not in excluded_ids
            ),
            key=lambda row: str(row["example_id"]),
        )
        if not candidates:
            raise ValueError(
                "No real-random donor for "
                f"recipient={recipient['example_id']} family={family_of(recipient)}"
            )
        controls.append(rng.choice(candidates))
    return controls


def parse_dose_alphas(spec: str | None) -> list[float]:
    if not spec:
        return []
    values = [float(value.strip()) for value in spec.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError("--dose-alphas contains duplicates")
    if any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("--dose-alphas must contain only strict interior values in (0, 1)")
    return sorted(values)


def dose_condition(alpha: float) -> str:
    return "dose_" + format(alpha, ".6g").replace(".", "p")


def interpolate_patch_vectors(
    recipient_vectors: dict[int, torch.Tensor],
    donor_vectors: dict[int, torch.Tensor],
    layers: list[int],
    alpha: float,
) -> dict[int, torch.Tensor]:
    return {
        layer: recipient_vectors[layer] + alpha * (donor_vectors[layer] - recipient_vectors[layer])
        for layer in layers
    }


def load_pair_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    task: str,
    swap_target: str,
    expected_pairs: int,
    allow_prefix: bool = False,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any] | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != task:
        raise ValueError(f"Pair manifest task={payload.get('task')!r}, expected {task!r}")
    if payload.get("swap_target") != swap_target:
        raise ValueError(
            f"Pair manifest swap_target={payload.get('swap_target')!r}, expected {swap_target!r}"
        )
    specs = payload.get("pairs") or []
    if allow_prefix:
        if len(specs) < expected_pairs:
            raise ValueError(
                f"Pair manifest has {len(specs)} pairs, fewer than requested prefix {expected_pairs}"
            )
        specs = specs[:expected_pairs]
    elif len(specs) != expected_pairs:
        raise ValueError(f"Pair manifest has {len(specs)} pairs, expected {expected_pairs}")
    by_id = {str(row["example_id"]): row for row in rows}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    controls: list[dict[str, Any] | None] = []
    for index, spec in enumerate(specs, start=1):
        donor_id = str(spec["donor_id"])
        recipient_id = str(spec["recipient_id"])
        samemal_id = spec.get("samemal_id")
        if donor_id not in by_id or recipient_id not in by_id:
            raise ValueError(f"Pair manifest row {index} references an unknown donor/recipient")
        donor, recipient = by_id[donor_id], by_id[recipient_id]
        if not valid_pair_for_target(donor, recipient, swap_target):
            raise ValueError(f"Pair manifest row {index} is no longer a valid pair")
        control = None
        if samemal_id is not None:
            samemal_id = str(samemal_id)
            if samemal_id not in by_id:
                raise ValueError(f"Pair manifest row {index} references unknown samemal_id={samemal_id}")
            control = by_id[samemal_id]
            if family_of(control) != family_of(recipient) or label_of(control) != label_of(recipient):
                raise ValueError(f"Pair manifest row {index} has a mismatched same-malrule control")
            if control["example_id"] == recipient["example_id"]:
                raise ValueError(f"Pair manifest row {index} uses recipient as its own control")
        pairs.append((donor, recipient))
        controls.append(control)
    return pairs, controls


def random_patch_vectors(
    source_vectors: dict[int, torch.Tensor],
    layers: list[int],
    seed: int,
) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        source = source_vectors[layer].float().cpu()
        vec = torch.randn(source.shape, generator=generator, dtype=torch.float32)
        vec_norm = torch.linalg.vector_norm(vec).clamp_min(1e-12)
        source_norm = torch.linalg.vector_norm(source)
        out[layer] = vec * (source_norm / vec_norm)
    return out


def load_model(
    model_path: str,
    adapter_path: str | None,
    bottleneck_mask: bool = False,
    state_token_alias: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source(model_path, adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    marker_text = state_token_alias or STATE_TOKEN
    marker_ids = tokenizer(marker_text, add_special_tokens=False)["input_ids"]
    if state_token_alias and len(marker_ids) != 1:
        raise ValueError(f"--state-token-alias must tokenize to one id, got {marker_ids}")
    print(
        f"[swap] marker={marker_text!r} marker_ids={marker_ids} "
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
            f"[swap] resize model embeddings {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}",
            flush=True,
        )
        model.resize_token_embeddings(len(tokenizer))
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def capture_state_vectors(
    model,
    tokenizer,
    row: dict[str, Any],
    layers: list[int],
    max_length: int,
    bottleneck_mask: bool = False,
    bottleneck_style: str = "standard",
    state_token_alias: str | None = None,
) -> dict[int, torch.Tensor]:
    if bottleneck_mask:
        ids, history_start, state_pos = locate_bottleneck_positions(
            tokenizer,
            str(row["prompt"]),
            state_token_alias,
        )
    else:
        ids, state_pos = locate_state_position(tokenizer, str(row["prompt"]), state_token_alias)
        history_start = -1
    if len(ids) > max_length:
        raise ValueError(f"Prompt too long for max_length={max_length}: {len(ids)}")
    device = getattr(model, "device", next(model.parameters()).device)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn_2d = torch.ones_like(input_ids)
    if bottleneck_mask:
        attn = build_bottleneck_attention_mask(
            attn_2d,
            torch.tensor([history_start], dtype=torch.long, device=device),
            torch.tensor([state_pos], dtype=torch.long, device=device),
            torch.tensor([len(ids)], dtype=torch.long, device=device),
            bottleneck_style,
        )
    else:
        attn = attn_2d
    decoder_layers = get_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0, state_pos].detach().float().cpu()

        return hook

    handles = [decoder_layers[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    try:
        _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def encode_candidate_batch(
    tokenizer,
    row: dict[str, Any],
    max_length: int,
    bottleneck_mask: bool = False,
    state_token_alias: str | None = None,
):
    prompt = str(row["prompt"])
    if bottleneck_mask:
        prefix_ids, history_start, state_pos = locate_bottleneck_positions(
            tokenizer,
            prompt,
            state_token_alias,
        )
    else:
        prefix_ids, state_pos = locate_state_position(tokenizer, prompt, state_token_alias)
        history_start = -1
    labels = list(row["candidate_labels"])
    items = []
    for label in labels:
        target = row["candidate_targets"][label]
        target_ids = tokenizer(str(target), add_special_tokens=False)["input_ids"]
        if len(prefix_ids) + len(target_ids) > max_length:
            raise ValueError(f"Candidate too long for max_length={max_length}")
        items.append(
            {
                "label": label,
                "input_ids": prefix_ids + target_ids,
                "prefix_len": len(prefix_ids),
                "target_len": len(target_ids),
            }
        )
    return items, history_start, state_pos


@torch.inference_mode()
def score_row(
    model,
    tokenizer,
    row: dict[str, Any],
    layers: list[int],
    patch_vectors: dict[int, torch.Tensor] | None,
    max_length: int,
    bottleneck_mask: bool = False,
    bottleneck_style: str = "standard",
    state_token_alias: str | None = None,
) -> dict[str, Any]:
    items, history_start, state_pos = encode_candidate_batch(
        tokenizer,
        row,
        max_length,
        bottleneck_mask,
        state_token_alias,
    )
    device = getattr(model, "device", next(model.parameters()).device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(item["input_ids"]) for item in items)
    input_ids, attn = [], []
    for item in items:
        ids = item["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    attn_tensor_2d = torch.tensor(attn, dtype=torch.long, device=device)
    if bottleneck_mask:
        attn_tensor = build_bottleneck_attention_mask(
            attn_tensor_2d,
            torch.full((len(items),), history_start, dtype=torch.long, device=device),
            torch.full((len(items),), state_pos, dtype=torch.long, device=device),
            torch.tensor([int(item["prefix_len"]) for item in items], dtype=torch.long, device=device),
            bottleneck_style,
        )
    else:
        attn_tensor = attn_tensor_2d

    handles = []
    if patch_vectors is not None:
        decoder_layers = get_decoder_layers(model)

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                vec = patch_vectors[layer_idx].to(patched.device, dtype=patched.dtype)
                patched[:, state_pos, :] = vec.unsqueeze(0)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            return hook

        handles = [decoder_layers[layer].register_forward_hook(make_hook(layer)) for layer in layers]

    try:
        logits = model(input_ids=input_tensor, attention_mask=attn_tensor, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()

    scores: dict[str, float] = {}
    for i, item in enumerate(items):
        ids = item["input_ids"]
        score = 0.0
        for pos in range(item["prefix_len"], len(ids)):
            logp = torch.log_softmax(logits[i, pos - 1, :].float(), dim=-1)
            score += float(logp[ids[pos]].item())
        scores[item["label"]] = score
    labels = list(scores)
    values = torch.tensor([scores[label] for label in labels], dtype=torch.float32)
    probs_tensor = torch.softmax(values, dim=0)
    probs = {label: float(prob) for label, prob in zip(labels, probs_tensor.tolist())}
    pred = max(scores, key=scores.get)
    return {"pred": pred, "scores": scores, "probs": probs}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)
    out = {}
    for cond, rows in sorted(by_condition.items()):
        categories = Counter(row["category"] for row in rows)
        out[cond] = {
            "n": len(rows),
            "category_counts": dict(categories),
            "category_rates": {k: v / len(rows) for k, v in categories.items()},
            "pred_is_A_rate": sum(row["pred"] == row.get("donor_target", row["donor_label"]) for row in rows)
            / len(rows),
            "pred_is_B_rate": sum(row["pred"] == row.get("recipient_target", row["recipient_label"]) for row in rows)
            / len(rows),
            "mean_p_A": sum(row["p_A"] for row in rows) / len(rows),
            "mean_p_B": sum(row["p_B"] for row in rows) / len(rows),
            "mean_margin_A_minus_B": sum(row["score_A"] - row["score_B"] for row in rows) / len(rows),
        }
    return out


def summarize_by_layer_config(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_config[str(record.get("layer_config", "default"))].append(record)
    return {config: summarize(rows) for config, rows in sorted(by_config.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--task", default="malrule_identity")
    parser.add_argument("--swap-target", choices=["identity", "response"], default="identity")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--scan-single-layers", action="store_true")
    parser.add_argument(
        "--layer-groups",
        help="Named semicolon-separated patch groups, e.g. late=48-63;final=63",
    )
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        help="Frozen donor/recipient/same-malrule pair manifest for reproducible causal comparison.",
    )
    parser.add_argument(
        "--allow-manifest-prefix",
        action="store_true",
        help="Diagnostic only: use the first --num-pairs rows of a longer frozen manifest.",
    )
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--bottleneck-mask", action="store_true")
    parser.add_argument("--bottleneck-style", choices=BOTTLENECK_STYLES, default="standard")
    parser.add_argument(
        "--state-token-alias",
        help=f"Token used in place of {STATE_TOKEN} during training, e.g. <|vision_pad|>.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--real-random-control",
        action="store_true",
        help="Add a seeded in-distribution same-family, different-rule donor control.",
    )
    parser.add_argument(
        "--real-random-seed",
        type=int,
        default=20_260_904,
        help="Seed used only to select real-random donor identities.",
    )
    parser.add_argument(
        "--dose-alphas",
        help="Comma-separated interior interpolation doses; noop is alpha=0 and swap is alpha=1.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl, args.task)
    if args.pair_manifest:
        pairs, samemal_controls = load_pair_manifest(
            args.pair_manifest,
            rows,
            args.task,
            args.swap_target,
            args.num_pairs,
            allow_prefix=args.allow_manifest_prefix,
        )
    else:
        pairs = build_pairs(rows, args.num_pairs, args.seed, args.swap_target)
        samemal_controls = build_samemal_controls(rows, pairs, args.seed + 1009)
    real_random_controls = (
        build_real_random_controls(rows, pairs, samemal_controls, args.real_random_seed)
        if args.real_random_control
        else [None] * len(pairs)
    )
    dose_alphas = parse_dose_alphas(args.dose_alphas)
    tokenizer, model = load_model(
        args.model,
        args.adapter,
        args.bottleneck_mask,
        args.state_token_alias,
    )
    n_layers = len(get_decoder_layers(model))
    layers = parse_layers(args.layers, n_layers)
    if args.layer_groups and args.scan_single_layers:
        raise ValueError("--layer-groups and --scan-single-layers are mutually exclusive")
    if args.layer_groups:
        layer_configs = parse_layer_groups(args.layer_groups, n_layers)
        layers = sorted({layer for _name, group in layer_configs for layer in group})
    elif args.scan_single_layers:
        if args.scan_stride < 1:
            raise ValueError("--scan-stride must be >= 1")
        layer_configs = [(f"L{layer}", [layer]) for layer in layers[:: args.scan_stride]]
    else:
        layer_configs = [(",".join(str(layer) for layer in layers) if args.layers != "all" else "all", layers)]
    capture_layers = sorted({layer for _name, config_layers in layer_configs for layer in config_layers})
    print(
        f"[swap] rows={len(rows)} pairs={len(pairs)} configs={len(layer_configs)} "
        f"capture_layers={capture_layers[:8]}... n_capture_layers={len(capture_layers)}",
        flush=True,
    )

    unique_rows: dict[str, dict[str, Any]] = {}
    for donor, recipient in pairs:
        unique_rows[donor["example_id"]] = donor
        unique_rows[recipient["example_id"]] = recipient
    for control in samemal_controls:
        if control is not None:
            unique_rows[control["example_id"]] = control
    for control in real_random_controls:
        if control is not None:
            unique_rows[control["example_id"]] = control

    state_cache: dict[str, dict[int, torch.Tensor]] = {}
    for idx, (example_id, row) in enumerate(unique_rows.items(), start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(unique_rows):
            print(f"[swap] capture state {idx}/{len(unique_rows)}", flush=True)
        state_cache[example_id] = capture_state_vectors(
            model,
            tokenizer,
            row,
            capture_layers,
            args.max_length,
            args.bottleneck_mask,
            args.bottleneck_style,
            args.state_token_alias,
        )

    records = []
    for config_idx, (layer_config_name, config_layers) in enumerate(layer_configs, start=1):
        print(
            f"[swap] score config {config_idx}/{len(layer_configs)} "
            f"{layer_config_name} layers={config_layers}",
            flush=True,
        )
        for pair_idx, ((donor, recipient), samemal_control, real_random_control) in enumerate(
            zip(pairs, samemal_controls, real_random_controls, strict=True), start=1
        ):
            if pair_idx == 1 or pair_idx % 5 == 0 or pair_idx == len(pairs):
                print(f"[swap] score pair {pair_idx}/{len(pairs)}", flush=True)
            donor_label = label_of(donor)
            recipient_label = label_of(recipient)
            donor_target, recipient_target, correct_target = pair_targets(donor, recipient, args.swap_target)
            conditions = {
                "none": None,
                "noop": state_cache[recipient["example_id"]],
                "swap": state_cache[donor["example_id"]],
                "random": random_patch_vectors(
                    state_cache[recipient["example_id"]],
                    config_layers,
                    args.seed + 1_000_003 * config_idx + pair_idx,
                ),
            }
            if samemal_control is not None:
                conditions["samemal"] = state_cache[samemal_control["example_id"]]
            if real_random_control is not None:
                conditions["random_real"] = state_cache[real_random_control["example_id"]]
            for alpha in dose_alphas:
                conditions[dose_condition(alpha)] = interpolate_patch_vectors(
                    state_cache[recipient["example_id"]],
                    state_cache[donor["example_id"]],
                    config_layers,
                    alpha,
                )
            for condition, patch_vectors in conditions.items():
                scored = score_row(
                    model,
                    tokenizer,
                    recipient,
                    config_layers,
                    patch_vectors,
                    args.max_length,
                    args.bottleneck_mask,
                    args.bottleneck_style,
                    args.state_token_alias,
                )
                pred = scored["pred"]
                if pred == donor_target:
                    category = "flip_to_A"
                elif pred == recipient_target:
                    category = "stay_B"
                elif correct_target is not None and pred == correct_target:
                    category = "correct"
                else:
                    category = "other"
                records.append(
                    {
                        "layer_config": layer_config_name,
                        "layers": config_layers,
                        "pair_idx": pair_idx,
                        "condition": condition,
                        "category": category,
                        "pred": pred,
                        "donor_id": donor["example_id"],
                        "recipient_id": recipient["example_id"],
                        "samemal_id": samemal_control["example_id"] if samemal_control is not None else None,
                        "donor_label": donor_label,
                        "recipient_label": recipient_label,
                        "samemal_label": label_of(samemal_control) if samemal_control is not None else None,
                        "real_random_id": (
                            real_random_control["example_id"] if real_random_control is not None else None
                        ),
                        "real_random_label": (
                            label_of(real_random_control) if real_random_control is not None else None
                        ),
                        "real_random_target_on_recipient_query": (
                            response_answer_for_malrule(recipient, label_of(real_random_control))
                            if real_random_control is not None
                            else None
                        ),
                        "donor_target": donor_target,
                        "recipient_target": recipient_target,
                        "correct_target": correct_target,
                        "family": family_of(recipient),
                        "score_A": scored["scores"].get(donor_target, float("-inf")),
                        "score_B": scored["scores"].get(recipient_target, float("-inf")),
                        "p_A": scored["probs"].get(donor_target, 0.0),
                        "p_B": scored["probs"].get(recipient_target, 0.0),
                        "top5": sorted(scored["scores"].items(), key=lambda kv: kv[1], reverse=True)[:5],
                    }
                )

    result = {
        "model": args.model,
        "adapter": args.adapter,
        "jsonl": str(args.jsonl),
        "task": args.task,
        "swap_target": args.swap_target,
        "layers": layers,
        "layer_configs": [{"name": name, "layers": config_layers} for name, config_layers in layer_configs],
        "bottleneck_mask": args.bottleneck_mask,
        "bottleneck_style": args.bottleneck_style if args.bottleneck_mask else None,
        "state_token_alias": args.state_token_alias,
        "pair_manifest": str(args.pair_manifest) if args.pair_manifest else None,
        "num_pairs": len(pairs),
        "real_random_control": args.real_random_control,
        "real_random_seed": args.real_random_seed if args.real_random_control else None,
        "real_random_selection": (
            "same_family; rule differs from directed donor and recipient; excludes donor, "
            "recipient, and same-rule row; sorted candidate pool then seeded choice; "
            "independent of model outputs"
            if args.real_random_control
            else None
        ),
        "dose_alphas": [0.0, *dose_alphas, 1.0] if dose_alphas else [],
        "dose_endpoint_aliases": {"0": "noop", "1": "swap"} if dose_alphas else {},
        "dose_formula": "h_alpha[layer] = (1-alpha) * h_recipient[layer] + alpha * h_donor[layer]",
        "summary": summarize(records) if len(layer_configs) == 1 else summarize_by_layer_config(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False), flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
