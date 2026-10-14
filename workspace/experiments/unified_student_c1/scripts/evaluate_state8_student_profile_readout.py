#!/usr/bin/env python3
"""End-to-end readable State8 profile and fixed-query transport evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_state8_student_profile_readout import NEUTRAL_TARGET, PROFILE_TASK, read_jsonl, write_jsonl
from evaluate_recurrent_causal_gate import random_state_like
from evaluate_recurrent_formal_protocols import generate_one, load_wrapper
from train_recurrent_student_m4 import rollout_state


TAG = {
    name: re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.I | re.S)
    for name in ("weakness", "strength", "profile", "evidence_confidence")
}


def release_inference_memory() -> None:
    """Release short-lived generation/KV objects before the next rollout."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def values(text: str, name: str) -> list[str]:
    match = TAG[name].search(str(text))
    if not match:
        return []
    value = match.group(1).strip()
    if value.lower() == "insufficient_evidence":
        return ["insufficient_evidence"]
    return [piece.strip() for piece in value.split(";") if piece.strip()]


def profile_key(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(values(text, "weakness")), tuple(values(text, "strength"))


def set_f1(predicted: list[str], gold: list[str]) -> float:
    p, g = set(predicted), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = len(p & g)
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def profile_similarity(predicted: str, target: str) -> float:
    return 0.5 * (
        set_f1(values(predicted, "weakness"), values(target, "weakness"))
        + set_f1(values(predicted, "strength"), values(target, "strength"))
    )


@torch.inference_mode()
def evaluate_profiles(wrapper, tokenizer, rows, alias, max_length, max_new_tokens):
    device = next(wrapper.parameters()).device
    records = []
    for index, row in enumerate(rows, start=1):
        state = rollout_state(wrapper, tokenizer, row["update_texts"], alias, max_length, device)
        output = generate_one(
            wrapper, tokenizer, state, row["decode_text"], alias, max_length, max_new_tokens
        )
        predicted_weak = values(output, "weakness")
        predicted_strong = values(output, "strength")
        records.append(
            {
                "profile_index": index,
                "output": output,
                "gold": row["target"],
                "predicted_weaknesses": predicted_weak,
                "predicted_strengths": predicted_strong,
                "observed_weakness_f1": set_f1(
                    predicted_weak, row["observed_profile"]["weaknesses"]
                ),
                "observed_strength_f1": set_f1(
                    predicted_strong, row["observed_profile"]["strengths"]
                ),
                "future_weakness_f1": set_f1(predicted_weak, row["future_weaknesses"]),
                "exact_profile": profile_key(output) == profile_key(row["target"]),
                "schema_valid": all(TAG[name].search(output) for name in TAG),
                "raw_history_visible_to_decoder": False,
                "student_id_visible_to_decoder": False,
            }
        )
        print(json.dumps({"profiles": index, "total": len(rows)}), flush=True)
        del state
        release_inference_memory()
    return records


@torch.inference_mode()
def evaluate_transport(wrapper, tokenizer, rows, alias, max_length, max_new_tokens, seed):
    device = next(wrapper.parameters()).device
    records = []
    for index, row in enumerate(rows, start=1):
        native = rollout_state(wrapper, tokenizer, row["native_update_texts"], alias, max_length, device)
        donor = rollout_state(wrapper, tokenizer, row["donor_update_texts"], alias, max_length, device)
        same = rollout_state(wrapper, tokenizer, row["same_rule_control_update_texts"], alias, max_length, device)
        states = {
            "native": native,
            "noop": native.clone(),
            "same_profile": same,
            "random": random_state_like(native, seed + 1_000_003 * index),
            "swap": donor,
        }
        native_key, donor_key = profile_key(row["native_target"]), profile_key(row["donor_target"])
        for condition, state in states.items():
            output = generate_one(
                wrapper, tokenizer, state, row["decode_text"], alias, max_length, max_new_tokens
            )
            key = profile_key(output)
            native_similarity = profile_similarity(output, row["native_target"])
            donor_similarity = profile_similarity(output, row["donor_target"])
            category = (
                "neutral"
                if key == profile_key(NEUTRAL_TARGET)
                else "donor"
                if donor_similarity > native_similarity and donor_similarity >= 0.5
                else "recipient"
                if native_similarity > donor_similarity and native_similarity >= 0.5
                else "other"
            )
            records.append(
                {
                    "pair_index": index,
                    "pair_id": row["pair_id"],
                    "condition": condition,
                    "category": category,
                    "native_profile_similarity": native_similarity,
                    "donor_profile_similarity": donor_similarity,
                    "output": output,
                    "native_target": row["native_target"],
                    "donor_target": row["donor_target"],
                    "query_held_fixed": True,
                    "only_state_identity_changes": True,
                }
            )
            release_inference_memory()
        print(json.dumps({"transport_pairs": index, "total": len(rows)}), flush=True)
        del states, native, donor, same
        release_inference_memory()
    return records


def summarize_transport(rows):
    by = defaultdict(list)
    for row in rows:
        by[row["condition"]].append(row)
    summary = {}
    for condition, values_ in sorted(by.items()):
        n = len(values_)
        summary[condition] = {
            "n": n,
            "to_donor_rate": sum(row["category"] == "donor" for row in values_) / n,
            "to_recipient_rate": sum(row["category"] == "recipient" for row in values_) / n,
            "neutral_rate": sum(row["category"] == "neutral" for row in values_) / n,
        }
    return {
        "by_condition": summary,
        "D_profile_specific": summary["swap"]["to_donor_rate"] - summary["random"]["to_donor_rate"],
        "D_profile_same_control": summary["native"]["to_recipient_rate"] - summary["same_profile"]["to_recipient_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--causal-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--num-profiles", type=int, default=88)
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    tokenizer, wrapper = load_wrapper(
        args.model, str(args.adapter), args.recurrent_state, args.state_token_alias, "layerwise"
    )
    if wrapper.state_token_count != 8:
        raise ValueError("Profile evaluation requires State8")
    alias = wrapper.state_token_rendered_alias
    profiles = evaluate_profiles(
        wrapper, tokenizer, read_jsonl(args.test_jsonl)[: args.num_profiles], alias,
        args.max_length, args.max_new_tokens,
    )
    transport = evaluate_transport(
        wrapper, tokenizer, read_jsonl(args.causal_jsonl)[: args.num_pairs], alias,
        args.max_length, args.max_new_tokens, args.seed,
    )
    profile_metrics = {
        "n": len(profiles),
        "schema_valid_rate": sum(row["schema_valid"] for row in profiles) / len(profiles),
        "exact_profile_accuracy": sum(row["exact_profile"] for row in profiles) / len(profiles),
        "observed_weakness_macro_f1": sum(row["observed_weakness_f1"] for row in profiles) / len(profiles),
        "observed_strength_macro_f1": sum(row["observed_strength_f1"] for row in profiles) / len(profiles),
        "future_weakness_macro_f1": sum(row["future_weakness_f1"] for row in profiles) / len(profiles),
    }
    transport_metrics = summarize_transport(transport)
    result = {
        "stage": "state8_future_student_profile_end_to_end_readout_v2",
        "status": "complete",
        "task": PROFILE_TASK,
        "claim": "State8 predicts a readable held-out future student profile from prefix history.",
        "swap_role": "diagnostic_only_not_a_scientific_gate",
        "state_token_count": 8,
        "state_shape": [64, 8, 5120],
        "profile": profile_metrics,
        "transport": transport_metrics,
        "query_held_fixed": True,
        "only_state_identity_changes": True,
        "raw_history_visible_to_decoder": False,
        "student_id_visible_to_decoder": False,
        "artifact_sha256": {
            "adapter_model": sha256(args.adapter / "adapter_model.safetensors"),
            "recurrent_state": sha256(args.recurrent_state),
        },
        "scientific_gate_pass": bool(
            profile_metrics["schema_valid_rate"] >= 0.95
            and profile_metrics["future_weakness_macro_f1"] >= 0.60
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "readable_profiles.jsonl", profiles)
    write_jsonl(args.output_dir / "fixed_query_transport.jsonl", transport)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
