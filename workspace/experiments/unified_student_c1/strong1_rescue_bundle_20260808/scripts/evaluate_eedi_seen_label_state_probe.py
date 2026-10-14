#!/usr/bin/env python3
"""Diagnostic Eedi T3 probe using train-only fitting and held-out formal rows."""

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
from probe_student_state_smoke import train_linear_probe


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def eedi_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("task") == "diagnose_misconception"
        and row.get("source") == "eedi_mining"
        and row.get("target_label") is not None
    ]


def select_panel(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    min_train_per_label: int,
    max_train_per_label: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    train_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eedi_rows(train_rows):
        train_buckets[str(row["target_label"])].append(row)
    for row in eedi_rows(eval_rows):
        eval_buckets[str(row["target_label"])].append(row)
    labels = sorted(
        label
        for label, rows in train_buckets.items()
        if len(rows) >= min_train_per_label and eval_buckets.get(label)
    )
    if len(labels) < 2:
        raise ValueError("Eedi state probe needs at least two eligible labels")
    selected_train: list[dict[str, Any]] = []
    selected_eval: list[dict[str, Any]] = []
    for label in labels:
        train = list(train_buckets[label])
        evaluation = list(eval_buckets[label])
        random.Random(f"{seed}:train:{label}").shuffle(train)
        random.Random(f"{seed}:eval:{label}").shuffle(evaluation)
        selected_train.extend(train[:max_train_per_label])
        selected_eval.extend(evaluation)
    random.Random(f"{seed}:train:final").shuffle(selected_train)
    random.Random(f"{seed}:eval:final").shuffle(selected_eval)
    return selected_train, selected_eval, labels


def stack(states: list[torch.Tensor | None]) -> np.ndarray:
    if any(state is None for state in states):
        raise ValueError("Probe requires a state for every row")
    return torch.stack([state for state in states if state is not None]).float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--min-train-per-label", type=int, default=5)
    parser.add_argument("--max-train-per-label", type=int, default=20)
    parser.add_argument("--state-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_train_per_label < 2 or args.max_train_per_label < args.min_train_per_label:
        raise ValueError("Invalid train-label support limits")

    train, evaluation, labels = select_panel(
        read_jsonl(args.train_jsonl),
        read_jsonl(args.eval_jsonl),
        min_train_per_label=args.min_train_per_label,
        max_train_per_label=args.max_train_per_label,
        seed=args.seed,
    )
    mapping = {label: index for index, label in enumerate(labels)}
    y_train = np.asarray([mapping[str(row["target_label"])] for row in train], dtype=np.int64)
    y_eval = np.asarray([mapping[str(row["target_label"])] for row in evaluation], dtype=np.int64)
    tokenizer, wrapper = load_wrapper(
        args.model,
        args.adapter,
        args.recurrent_state,
        args.state_token_alias,
        "layerwise",
    )
    x_train = stack(
        recurrent_states_batched(
            wrapper,
            tokenizer,
            train,
            args.state_token_alias,
            args.max_length,
            args.state_batch_size,
        )
    )
    x_eval = stack(
        recurrent_states_batched(
            wrapper,
            tokenizer,
            evaluation,
            args.state_token_alias,
            args.max_length,
            args.state_batch_size,
        )
    )
    if x_train.ndim != 3 or x_train.shape[1:] != (64, 5120):
        raise ValueError(f"Unexpected layerwise state shape: {x_train.shape}")

    by_layer: dict[str, Any] = {}
    for layer in range(x_train.shape[1]):
        by_layer[str(layer)] = train_linear_probe(
            x_train[:, layer],
            y_train,
            x_eval[:, layer],
            y_eval,
            args.seed,
            args.epochs,
            args.lr,
            args.weight_decay,
        )
    best = max(
        by_layer,
        key=lambda layer: (by_layer[layer]["accuracy"], by_layer[layer]["macro_f1"]),
    )
    result = {
        "stage": "eedi_seen_label_layerwise_state_probe_v1",
        "diagnostic_only": True,
        "formal_t3_generation_still_required": True,
        "fit_split": "train_plus_validation_only",
        "evaluation_split": "frozen_official_fold_4",
        "raw_history_visible_to_probe": False,
        "persistent_history_kv_cache": False,
        "state_shape": [64, 5120],
        "label_selection": {
            "minimum_train_rows": args.min_train_per_label,
            "requires_formal_support": True,
            "selected_by_accuracy": False,
        },
        "n_labels": len(labels),
        "random_level": 1.0 / len(labels),
        "support": {"train": len(train), "evaluation": len(evaluation)},
        "train_label_counts": dict(Counter(str(row["target_label"]) for row in train)),
        "evaluation_label_counts": dict(
            Counter(str(row["target_label"]) for row in evaluation)
        ),
        "layers": by_layer,
        "best_layer": int(best),
        "best": by_layer[best],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_json)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
