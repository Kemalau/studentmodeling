#!/usr/bin/env python3
"""Frozen C4 gate for a zero-training layerwise recurrent state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_layerwise_state_bridge import activate_decode_adapter
from evaluate_recurrent_causal_gate import (
    random_state_like,
    summarize_records,
)
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from recurrent_student_model import build_update_text, extract_direct_decode_text
from state_swap_misconception_smoke import (
    family_of,
    label_of,
    load_model,
    load_pair_manifest,
    pair_targets,
    read_jsonl,
)
from train_recurrent_student_diagnostic import (
    alias_text,
    encode_prompt,
    encode_prompt_target,
    move,
    sequence_logprob,
)


@torch.inference_mode()
def state_for_row(
    wrapper,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
):
    state = wrapper.initial_state(1)
    history = list(row.get("history") or [])
    if not history:
        raise ValueError(f"C4 row has no observed history: {row.get('example_id')}")
    for interaction in history:
        update = move(
            encode_prompt(
                tokenizer,
                alias_text(build_update_text(interaction), alias),
                max_length,
            ),
            next(wrapper.parameters()).device,
        )
        state = wrapper.update_state(state, **update).state
    return state


@torch.inference_mode()
def score_state(
    wrapper,
    tokenizer,
    state,
    row,
    alias: str,
    max_length: int,
) -> dict[str, Any]:
    text = extract_direct_decode_text(str(row["prompt"]))
    if "<student_history>" in text.lower():
        raise AssertionError("Layerwise recurrent decoder leaked raw history")
    device = next(wrapper.parameters()).device
    scores = {}
    for label_value in row["candidate_labels"]:
        label = str(label_value)
        encoded = move(
            encode_prompt_target(
                tokenizer,
                alias_text(text, alias),
                str(row["candidate_targets"][label]),
                max_length,
            ),
            device,
        )
        prompt_end = encoded.pop("prompt_ends")
        output = wrapper.decode_from_state(
            state, **encoded, prompt_ends=prompt_end
        )
        scores[label] = sequence_logprob(
            output.logits,
            encoded["input_ids"],
            int(prompt_end.item()),
        )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--decode-adapter")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
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
    tokenizer, backbone = load_model(
        args.model,
        args.adapter,
        bottleneck_mask=True,
        state_token_alias=args.state_token_alias,
    )
    state_ids = tokenizer(
        args.state_token_alias, add_special_tokens=False
    )["input_ids"]
    if len(state_ids) != 1:
        raise ValueError(f"State token alias is not atomic: {state_ids}")
    wrapper = LayerwiseRecurrentStudentModel(backbone, int(state_ids[0]))
    wrapper.eval()
    unique: dict[str, dict[str, Any]] = {}
    for donor, recipient in pairs:
        unique[str(donor["example_id"])] = donor
        unique[str(recipient["example_id"])] = recipient
    for control in controls:
        if control is not None:
            unique[str(control["example_id"])] = control
    state_cache = {}
    for index, (example_id, row) in enumerate(unique.items(), start=1):
        state_cache[example_id] = state_for_row(
            wrapper, tokenizer, row, args.state_token_alias, args.max_length
        )
        if index == 1 or index % 10 == 0 or index == len(unique):
            print(json.dumps({"states": index, "total": len(unique)}), flush=True)

    decode_adapter = args.decode_adapter or args.adapter
    active_adapter = activate_decode_adapter(
        backbone, args.adapter, decode_adapter
    )
    records = []
    for pair_index, ((donor, recipient), same_control) in enumerate(
        zip(pairs, controls, strict=True), start=1
    ):
        donor_target, recipient_target, _ = pair_targets(
            donor, recipient, "response"
        )
        recipient_state = state_cache[str(recipient["example_id"])]
        conditions = {
            "native": recipient_state,
            "noop": recipient_state.clone(),
            "swap": state_cache[str(donor["example_id"])],
            "random": random_state_like(
                recipient_state, args.seed + 1_000_003 * pair_index
            ),
            "samemal": state_cache[str(same_control["example_id"])],
        }
        for condition, state in conditions.items():
            scored = score_state(
                wrapper,
                tokenizer,
                state,
                recipient,
                args.state_token_alias,
                args.max_length,
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
    result = {
        "model": args.model,
        "capture_adapter": args.adapter,
        "decode_adapter": decode_adapter,
        "active_decode_adapter": active_adapter,
        "jsonl": str(args.jsonl),
        "pair_manifest": str(args.pair_manifest),
        "intervention_target": "final_layerwise_recurrent_state",
        "state_shape": [wrapper.layer_count, wrapper.hidden_size],
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
