#!/usr/bin/env python3
"""Frozen C4a linear probe over the actual closed recurrent student state."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_recurrent_formal_protocols import load_wrapper, recurrent_states_batched
from probe_student_state_smoke import macro_f1, train_linear_probe
from state_swap_misconception_smoke import label_of, read_jsonl


TASK = "malrule_next_response"


def stratified(rows: list[dict[str, Any]], per_label: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[label_of(row)].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in sorted(buckets):
        items = list(buckets[label])
        rng.shuffle(items)
        selected.extend(items[:per_label])
    rng.shuffle(selected)
    return selected


def labels_for(train: list[dict[str, Any]], *panels: list[dict[str, Any]]):
    labels = sorted({label_of(row) for row in train})
    mapping = {label: index for index, label in enumerate(labels)}
    unknown = sorted({label_of(row) for panel in panels for row in panel} - set(mapping))
    if unknown:
        raise ValueError(f"Probe evaluation has unseen labels: {unknown}")
    return labels, [
        np.asarray([mapping[label_of(row)] for row in rows], dtype=np.int64)
        for rows in (train, *panels)
    ]


def stack(states: list[torch.Tensor | None]) -> np.ndarray:
    if any(state is None for state in states):
        raise ValueError("C4a requires a non-empty recurrent state for every row")
    return torch.stack([state for state in states if state is not None]).float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--same-jsonl", type=Path, required=True)
    parser.add_argument("--cross-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--train-per-label", type=int, default=25)
    parser.add_argument("--eval-per-label", type=int, default=10)
    parser.add_argument("--state-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_per_label <= 0 or args.eval_per_label <= 0:
        raise ValueError("Per-label support must be positive")

    train = stratified(read_jsonl(args.train_jsonl, TASK), args.train_per_label, args.seed)
    same = stratified(read_jsonl(args.same_jsonl, TASK), args.eval_per_label, args.seed + 1)
    cross = stratified(read_jsonl(args.cross_jsonl, TASK), args.eval_per_label, args.seed + 2)
    labels, encoded = labels_for(train, same, cross)
    y_train, y_same, y_cross = encoded
    tokenizer, wrapper = load_wrapper(
        args.model, args.adapter, args.recurrent_state,
        args.state_token_alias, "layerwise",
    )
    x_train = stack(recurrent_states_batched(
        wrapper, tokenizer, train, args.state_token_alias, args.max_length, args.state_batch_size
    ))
    x_same = stack(recurrent_states_batched(
        wrapper, tokenizer, same, args.state_token_alias, args.max_length, args.state_batch_size
    ))
    x_cross = stack(recurrent_states_batched(
        wrapper, tokenizer, cross, args.state_token_alias, args.max_length, args.state_batch_size
    ))
    if x_train.ndim != 3 or x_train.shape[1:] != (64, 5120):
        raise ValueError(f"Unexpected layerwise recurrent state shape: {x_train.shape}")

    by_layer: dict[str, Any] = {}
    for layer in range(x_train.shape[1]):
        same_result = train_linear_probe(
            x_train[:, layer], y_train, x_same[:, layer], y_same,
            args.seed, args.epochs, args.lr, args.weight_decay,
        )
        cross_result = train_linear_probe(
            x_train[:, layer], y_train, x_cross[:, layer], y_cross,
            args.seed, args.epochs, args.lr, args.weight_decay,
        )
        by_layer[str(layer)] = {"same": same_result, "cross": cross_result}
    best = max(
        by_layer,
        key=lambda layer: (
            by_layer[layer]["cross"]["accuracy"],
            by_layer[layer]["cross"]["macro_f1"],
        ),
    )
    result = {
        "stage": "layerwise_recurrent_C4a_linear_probe",
        "state_architecture": "layerwise",
        "state_shape": [64, 5120],
        "raw_history_visible_to_probe": False,
        "persistent_history_kv_cache": False,
        "task": TASK,
        "n_labels": len(labels),
        "labels": labels,
        "random_level": 1.0 / len(labels),
        "support": {"train": len(train), "same": len(same), "cross": len(cross)},
        "label_counts": {
            "train": dict(Counter(label_of(row) for row in train)),
            "same": dict(Counter(label_of(row) for row in same)),
            "cross": dict(Counter(label_of(row) for row in cross)),
        },
        "layers": by_layer,
        "best_cross_layer": int(best),
        "best_cross": by_layer[best]["cross"],
        "scientific_gate": {
            "minimum_cross_accuracy": 0.50,
            "minimum_chance_multiple": 5.0,
            "pass": (
                by_layer[best]["cross"]["accuracy"] >= 0.50
                and by_layer[best]["cross"]["accuracy"] >= 5.0 / len(labels)
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["scientific_gate"]["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
