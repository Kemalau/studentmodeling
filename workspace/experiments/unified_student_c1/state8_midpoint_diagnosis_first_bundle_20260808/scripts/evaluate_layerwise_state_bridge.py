#!/usr/bin/env python3
"""Frozen C4 test of a closed layerwise state-address bridge.

The one-shot model writes a representation at one token address at every
decoder block.  A scalar recurrent state currently keeps only the final block
and reinjects it at the embedding layer, which mixes two representation spaces.
This diagnostic captures the complete residual trajectory at the one-shot
state address and patches that trajectory into a decoder that contains no raw
student history.  It executes no optimizer step.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from recurrent_student_model import (
    build_closed_decode_attention_mask,
    extract_direct_decode_text,
)
from state_swap_misconception_smoke import (
    build_bottleneck_attention_mask,
    encode_prompt_text,
    family_of,
    get_decoder_layers,
    label_of,
    load_model,
    load_pair_manifest,
    locate_bottleneck_positions,
    locate_state_position,
    pair_targets,
    random_patch_vectors,
    read_jsonl,
)


CONDITIONS = ("noop", "samemal", "random", "swap")
POSITION_MODES = ("compact", "one_shot_aligned")


def activate_decode_adapter(model, capture_adapter: str, decode_adapter: str) -> str:
    """Load a distinct decoder LoRA without reloading the 32B base model."""
    if decode_adapter == capture_adapter:
        return "default"
    if not hasattr(model, "load_adapter") or not hasattr(model, "set_adapter"):
        raise TypeError("Cross-adapter bridge requires a PEFT model")
    name = "closed_state_decoder"
    model.load_adapter(decode_adapter, adapter_name=name, is_trainable=False)
    model.set_adapter(name)
    model.eval()
    return name


def closed_position_ids(
    sequence_length: int,
    closed_state_position: int,
    one_shot_state_position: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Build compact or one-shot-aligned positions for a closed sequence."""
    if mode not in POSITION_MODES:
        raise ValueError(f"Unknown position mode: {mode}")
    if not (0 <= closed_state_position < sequence_length):
        raise ValueError("Closed state position is outside the sequence")
    if one_shot_state_position < closed_state_position:
        raise ValueError(
            "One-shot state position cannot precede the closed state position"
        )
    positions = torch.arange(sequence_length, dtype=torch.long, device=device)
    if mode == "one_shot_aligned":
        positions[closed_state_position:] += (
            one_shot_state_position - closed_state_position
        )
    return positions.unsqueeze(0)


@torch.inference_mode()
def capture_layerwise_state(
    model,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
) -> tuple[dict[int, torch.Tensor], int]:
    """Capture the residual trajectory at the one-shot state address."""
    prompt = str(row["prompt"])
    ids, history_start, state_position = locate_bottleneck_positions(
        tokenizer, prompt, alias
    )
    if len(ids) > max_length:
        raise ValueError(f"Prompt exceeds max_length={max_length}: {len(ids)}")
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_2d = torch.ones_like(input_ids)
    attention = build_bottleneck_attention_mask(
        attention_2d,
        torch.tensor([history_start], dtype=torch.long, device=device),
        torch.tensor([state_position], dtype=torch.long, device=device),
        torch.tensor([len(ids)], dtype=torch.long, device=device),
        "late_fusion",
    )
    layers = get_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = (
                hidden[0, state_position].detach().float().cpu()
            )

        return hook

    handles = [
        layer.register_forward_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(range(len(layers))):
        raise ValueError("Layerwise capture is incomplete")
    return captured, state_position


def encode_closed_candidates(
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
) -> tuple[list[dict[str, Any]], int]:
    text = extract_direct_decode_text(str(row["prompt"]))
    if "<student_history>" in text.lower():
        raise AssertionError("Closed bridge decoder leaked raw history")
    aliased = encode_prompt_text(text, alias)
    prefix_ids, state_position = locate_state_position(tokenizer, text, alias)
    if prefix_ids != tokenizer(aliased, add_special_tokens=False)["input_ids"]:
        raise AssertionError("Closed prompt tokenization drifted")
    items = []
    for label_value in row["candidate_labels"]:
        label = str(label_value)
        target_ids = tokenizer(
            str(row["candidate_targets"][label]), add_special_tokens=False
        )["input_ids"]
        ids = prefix_ids + target_ids
        if len(ids) > max_length:
            raise ValueError(f"Candidate exceeds max_length={max_length}")
        items.append(
            {
                "label": label,
                "input_ids": ids,
                "prompt_end": len(prefix_ids),
            }
        )
    return items, state_position


@torch.inference_mode()
def score_layerwise_state(
    model,
    tokenizer,
    state: dict[int, torch.Tensor],
    one_shot_state_position: int,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    position_mode: str,
) -> dict[str, Any]:
    items, state_position = encode_closed_candidates(
        tokenizer, row, alias, max_length
    )
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    maximum = max(len(item["input_ids"]) for item in items)
    input_ids = torch.tensor(
        [
            item["input_ids"] + [int(pad_id)] * (maximum - len(item["input_ids"]))
            for item in items
        ],
        dtype=torch.long,
        device=device,
    )
    attention_2d = torch.tensor(
        [
            [1] * len(item["input_ids"])
            + [0] * (maximum - len(item["input_ids"]))
            for item in items
        ],
        dtype=torch.long,
        device=device,
    )
    prompt_ends = torch.tensor(
        [item["prompt_end"] for item in items], dtype=torch.long, device=device
    )
    carrier = torch.full(
        (len(items),), state_position, dtype=torch.long, device=device
    )
    attention = build_closed_decode_attention_mask(
        attention_2d, carrier, prompt_ends
    )
    positions = closed_position_ids(
        maximum,
        state_position,
        one_shot_state_position,
        position_mode,
        device,
    ).expand(len(items), -1)
    layers = get_decoder_layers(model)
    if set(state) != set(range(len(layers))):
        raise ValueError("Layerwise state does not cover every decoder block")

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            vector = state[layer_index].to(
                device=patched.device, dtype=patched.dtype
            )
            patched[:, state_position, :] = vector.unsqueeze(0)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        return hook

    handles = [
        layer.register_forward_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
        ).logits
    finally:
        for handle in handles:
            handle.remove()
    scores: dict[str, float] = {}
    for row_index, item in enumerate(items):
        score = 0.0
        for token_position in range(item["prompt_end"], len(item["input_ids"])):
            log_probabilities = torch.log_softmax(
                logits[row_index, token_position - 1].float(), dim=-1
            )
            score += float(
                log_probabilities[input_ids[row_index, token_position]].item()
            )
        scores[item["label"]] = score
    labels = list(scores)
    probabilities = torch.softmax(
        torch.tensor([scores[label] for label in labels]), dim=0
    ).tolist()
    return {
        "prediction": max(scores, key=scores.get),
        "scores": scores,
        "probabilities": dict(zip(labels, probabilities, strict=True)),
    }


def categorize(prediction: str, donor_target: str, recipient_target: str) -> str:
    if prediction == donor_target:
        return "flip_to_A"
    if prediction == recipient_target:
        return "stay_B"
    return "other"


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_pair[int(record["pair_idx"])][str(record["condition"])] = record
    required = {"native", *CONDITIONS}
    incomplete = [
        pair_index
        for pair_index, conditions in by_pair.items()
        if set(conditions) != required
    ]
    if incomplete:
        raise ValueError(f"Incomplete bridge conditions: {incomplete[:5]}")
    clean_ids = sorted(
        pair_index
        for pair_index, conditions in by_pair.items()
        if conditions["native"]["category"] == "stay_B"
    )
    if not clean_ids:
        raise ValueError("Layerwise bridge produced no clean native=B pairs")
    native = [by_pair[index]["native"] for index in clean_ids]
    summary: dict[str, Any] = {
        "clean_definition": "native.category == stay_B",
        "all_pairs_n": len(by_pair),
        "clean_n": len(clean_ids),
        "clean": {},
    }
    for condition in CONDITIONS:
        rows = [by_pair[index][condition] for index in clean_ids]
        counts = Counter(row["category"] for row in rows)
        n = len(rows)
        summary["clean"][condition] = {
            "n": n,
            "category_counts": dict(sorted(counts.items())),
            "flip_to_A_rate": counts["flip_to_A"] / n,
            "stay_B_rate": counts["stay_B"] / n,
            "other_rate": counts["other"] / n,
            "changed_top1_rate": sum(
                row["prediction"] != base["prediction"]
                for row, base in zip(rows, native, strict=True)
            )
            / n,
            "mean_delta_margin_A_minus_B": sum(
                row["margin_A_minus_B"] - base["margin_A_minus_B"]
                for row, base in zip(rows, native, strict=True)
            )
            / n,
        }
    swap = summary["clean"]["swap"]["flip_to_A_rate"]
    random = summary["clean"]["random"]["flip_to_A_rate"]
    noop = summary["clean"]["noop"]["flip_to_A_rate"]
    summary["D_specific"] = swap - random
    summary["D_damage"] = random - noop
    checks = {
        "candidate_support": len(clean_ids) >= 50,
        "specificity": summary["D_specific"] >= 0.15,
        "noop_control": noop <= 0.05,
        "same_rule_control": summary["clean"]["samemal"]["flip_to_A_rate"]
        <= 0.10,
    }
    summary["checks"] = checks
    summary["scientific_gate_pass"] = all(checks.values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--decode-adapter")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--position-mode", choices=POSITION_MODES, required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--num-pairs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl, "malrule_next_response")
    pairs, controls = load_pair_manifest(
        args.pair_manifest,
        rows,
        "malrule_next_response",
        "response",
        args.num_pairs,
    )
    tokenizer, model = load_model(
        args.model,
        args.adapter,
        bottleneck_mask=True,
        state_token_alias=args.state_token_alias,
    )
    unique: dict[str, dict[str, Any]] = {}
    for donor, recipient in pairs:
        unique[str(donor["example_id"])] = donor
        unique[str(recipient["example_id"])] = recipient
    for control in controls:
        if control is not None:
            unique[str(control["example_id"])] = control
    state_cache: dict[str, dict[int, torch.Tensor]] = {}
    position_cache: dict[str, int] = {}
    for index, (example_id, row) in enumerate(unique.items(), start=1):
        state_cache[example_id], position_cache[example_id] = (
            capture_layerwise_state(
                model, tokenizer, row, args.state_token_alias, args.max_length
            )
        )
        if index == 1 or index % 10 == 0 or index == len(unique):
            print(json.dumps({"states": index, "total": len(unique)}), flush=True)

    decode_adapter = args.decode_adapter or args.adapter
    active_adapter = activate_decode_adapter(
        model, args.adapter, decode_adapter
    )

    records = []
    for pair_index, ((donor, recipient), same_control) in enumerate(
        zip(pairs, controls, strict=True), start=1
    ):
        donor_target, recipient_target, _ = pair_targets(
            donor, recipient, "response"
        )
        recipient_id = str(recipient["example_id"])
        conditions = {
            "native": state_cache[recipient_id],
            "noop": {k: v.clone() for k, v in state_cache[recipient_id].items()},
            "swap": state_cache[str(donor["example_id"])],
            "random": random_patch_vectors(
                state_cache[recipient_id],
                sorted(state_cache[recipient_id]),
                args.seed + 1_000_003 * pair_index,
            ),
            "samemal": state_cache[str(same_control["example_id"])],
        }
        for condition, state in conditions.items():
            scored = score_layerwise_state(
                model,
                tokenizer,
                state,
                # Keep the recipient's address fixed in every intervention.
                # Using the donor's original position would confound state
                # content with a condition-dependent RoPE offset.
                position_cache[recipient_id],
                recipient,
                args.state_token_alias,
                args.max_length,
                args.position_mode,
            )
            p_a = float(scored["probabilities"].get(donor_target, 0.0))
            p_b = float(scored["probabilities"].get(recipient_target, 0.0))
            score_a = float(scored["scores"].get(donor_target, float("-inf")))
            score_b = float(
                scored["scores"].get(recipient_target, float("-inf"))
            )
            records.append(
                {
                    "pair_idx": pair_index,
                    "condition": condition,
                    "category": categorize(
                        str(scored["prediction"]), donor_target, recipient_target
                    ),
                    "prediction": scored["prediction"],
                    "donor_id": donor["example_id"],
                    "recipient_id": recipient["example_id"],
                    "samemal_id": same_control["example_id"],
                    "donor_label": label_of(donor),
                    "recipient_label": label_of(recipient),
                    "family": family_of(recipient),
                    "donor_target": donor_target,
                    "recipient_target": recipient_target,
                    "p_A": p_a,
                    "p_B": p_b,
                    "margin_A_minus_B": score_a - score_b,
                }
            )
        if pair_index == 1 or pair_index % 10 == 0:
            print(json.dumps({"pairs": pair_index, "total": len(pairs)}), flush=True)
    layers = len(get_decoder_layers(model))
    hidden = int(next(iter(state_cache.values()))[0].numel())
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "capture_adapter": args.adapter,
        "decode_adapter": decode_adapter,
        "active_decode_adapter": active_adapter,
        "jsonl": str(args.jsonl),
        "pair_manifest": str(args.pair_manifest),
        "position_mode": args.position_mode,
        "intervention_target": "one_address_layerwise_residual_trajectory",
        "state_shape": [layers, hidden],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "optimizer_steps": 0,
        "summary": summarize_records(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
