#!/usr/bin/env python3
"""Bounded early audit for a State8 profile-ownership checkpoint.

This diagnostic uses only the frozen auxiliary ownership head stored in the
checkpoint.  It never substitutes for the preregistered reusable-interface
evaluation, but it can quickly show whether student-specific future-rate
information is beginning to emerge before the full run completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_recurrent_formal_protocols import pad_prompt_batches
from evaluate_state8_reusable_interface import canonical_features, update_texts
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from train_recurrent_student_diagnostic import alias_text, encode_prompt, move


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_wrapper(model: Path, adapter: Path, checkpoint: Path, alias: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    backbone = PeftModel.from_pretrained(backbone, adapter)
    ids = tokenizer(alias, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"State alias must be one token, got {ids}")
    wrapper = LayerwiseRecurrentStudentModel(
        backbone,
        int(ids[0]),
        transition_rank=int(config["layerwise_transition_rank"]),
        state_token_count=int(config["state_token_count"]),
    )
    wrapper.transition_in.load_state_dict(payload["transition_in"])
    wrapper.transition_out.load_state_dict(payload["transition_out"])
    device = backbone.get_input_embeddings().weight.device
    wrapper.transition_in.to(device=device)
    wrapper.transition_out.to(device=device)
    wrapper.state_token_rendered_alias = alias * int(config["state_token_count"])
    wrapper.eval()
    return tokenizer, wrapper, payload


@torch.inference_mode()
def states(wrapper, tokenizer, rows: list[dict], max_length: int, batch_size: int):
    device = wrapper.backbone.get_input_embeddings().weight.device
    pad_id = int(tokenizer.pad_token_id)
    updates = [update_texts(row, "full") for row in rows]
    by_depth: dict[int, list[int]] = {}
    for index, values in enumerate(updates):
        by_depth.setdefault(len(values), []).append(index)
    output: list[np.ndarray | None] = [None] * len(rows)
    completed = 0
    for depth, indices in sorted(by_depth.items()):
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            state = wrapper.initial_state(len(selected), device=device)
            for position in range(depth):
                encoded = [
                    move(
                        encode_prompt(
                            tokenizer,
                            alias_text(updates[index][position], wrapper.state_token_rendered_alias),
                            max_length,
                        ),
                        device,
                    )
                    for index in selected
                ]
                batch = pad_prompt_batches(encoded, pad_id)
                state = wrapper.update_state(state, **batch).state
            values = canonical_features(state, 64, 20260818)
            for index, value in zip(selected, values, strict=True):
                output[index] = value
            completed += len(selected)
            print(json.dumps({"states": completed, "total": len(rows)}), flush=True)
            del state, encoded, batch, values
            torch.cuda.empty_cache()
    if any(value is None for value in output):
        raise AssertionError("State extraction was incomplete")
    return np.stack([value for value in output if value is not None])


def masked_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.square(prediction[mask] - target[mask]).mean()))


def normalize_profile_features(features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return features
    if mode != "layernorm":
        raise ValueError(f"Unknown profile feature normalization: {mode}")
    mean = features.mean(axis=1, keepdims=True)
    variance = np.square(features - mean).mean(axis=1, keepdims=True)
    return (features - mean) / np.sqrt(variance + 1e-5)


def observed_future_mask(rows: list[dict], scope: str) -> np.ndarray:
    """Freeze the audit stratum from future support and updater-visible KCs."""
    future = np.asarray([row["future_eligible_mask"] for row in rows], dtype=bool)
    if scope == "all":
        return future
    if scope != "seen":
        raise ValueError(f"Unknown ownership scope: {scope}")
    observed = np.zeros_like(future)
    for row_index, row in enumerate(rows):
        vocabulary = [str(value) for value in row["kc_vocabulary"]]
        index = {name: position for position, name in enumerate(vocabulary)}
        for event in row["state_history"]:
            for name in set(str(value) for value in event["kc_names"]):
                if name in index:
                    observed[row_index, index[name]] = True
    return future & observed


def checkpoint_step(payload: dict) -> int:
    """Recover the audited micro-step from an immutable checkpoint contract.

    Intermediate optimizer checkpoints store ``step`` at the payload root.
    The trainer's terminal recurrent-state artifact intentionally stores only
    recurrent modules and its frozen run config, so bounded diagnostics record
    their exact terminal step as ``diagnostic_max_micro_steps`` instead.  Do
    not guess a value when neither contract is present.
    """
    value = payload.get("step")
    if value is None:
        value = (payload.get("config") or {}).get("diagnostic_max_micro_steps")
    if value is None or isinstance(value, bool):
        raise ValueError("Checkpoint does not record an auditable micro-step")
    step = int(value)
    if step <= 0:
        raise ValueError(f"Checkpoint micro-step must be positive, got {step}")
    return step


def ownership_permutation_null(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Compare native ownership with student-label permutations on held-out rows."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target, and mask must have identical shapes")
    if prediction.ndim != 2 or len(prediction) < 2:
        raise ValueError("Ownership null requires at least two aligned students")
    if replicates <= 0 or not bool(mask.any()):
        raise ValueError("Permutation replicates and supported cells must be positive")
    native = masked_rmse(prediction, target, mask)
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    identity = np.arange(len(prediction))
    for index in range(replicates):
        permutation = rng.permutation(len(prediction))
        if np.array_equal(permutation, identity):
            permutation = np.roll(permutation, 1)
        null[index] = masked_rmse(prediction[permutation], target, mask)
    return {
        "native_rmse": native,
        "permuted_rmse_mean": float(null.mean()),
        "permuted_rmse_ci95": [
            float(value) for value in np.quantile(null, [0.025, 0.975])
        ],
        "delta_permuted_mean_minus_native": float(null.mean() - native),
        "p_one_sided": float((1 + np.sum(null <= native)) / (replicates + 1)),
        "replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--permutation-replicates", type=int, default=2000)
    parser.add_argument("--ownership-scope", choices=("all", "seen"), default="all")
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    tokenizer, wrapper, payload = load_wrapper(
        args.model, args.adapter, args.checkpoint, args.state_token_alias
    )
    features = states(wrapper, tokenizer, rows, args.max_length, args.batch_size)
    head = payload["profile_ownership_head"]
    normalization = str(
        payload.get("config", {}).get("profile_ownership_feature_normalization")
        or "none"
    )
    features = normalize_profile_features(features, normalization)
    prediction = 1.0 / (1.0 + np.exp(-(features @ head["weight"].numpy().T + head["bias"].numpy())))
    target = np.asarray(
        [
            [0.5 if value is None else float(value) for value in row["future_smoothed_rates"]]
            for row in rows
        ],
        dtype=np.float32,
    )
    mask = observed_future_mask(rows, args.ownership_scope)
    ownership = ownership_permutation_null(
        prediction,
        target,
        mask,
        replicates=args.permutation_replicates,
        seed=args.seed,
    )
    result = {
        "stage": "state8_profile_ownership_checkpoint_early_audit_v1",
        "status": "complete",
        "diagnostic_only": True,
        "checkpoint_step": checkpoint_step(payload),
        "students": len(rows),
        "manifest_sha256": sha256(args.rows),
        "split": str(rows[0].get("split") or "unknown") if rows else "unknown",
        "supported_student_kc_cells": int(mask.sum()),
        "ownership_scope": args.ownership_scope,
        "native_rmse": ownership["native_rmse"],
        "permuted_ownership_rmse_mean": ownership["permuted_rmse_mean"],
        "permuted_ownership_rmse_ci95": ownership["permuted_rmse_ci95"],
        "ownership_delta_permuted_mean_minus_native": ownership[
            "delta_permuted_mean_minus_native"
        ],
        "ownership_permutation_p_one_sided": ownership["p_one_sided"],
        "ownership_permutation_replicates": ownership["replicates"],
        "artifact_sha256": {
            "adapter_model": sha256(args.adapter / "adapter_model.safetensors"),
            "recurrent_state": sha256(args.checkpoint),
        },
        "ownership_direction_positive": (
            ownership["delta_permuted_mean_minus_native"] > 0
        ),
        "strict_state_only": bool(
            payload["config"].get("final_inference_route") == "strict_state_only"
            and payload["config"].get("raw_history_visible_to_decoder") is False
        ),
        "profile_ownership_feature_normalization": normalization,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
