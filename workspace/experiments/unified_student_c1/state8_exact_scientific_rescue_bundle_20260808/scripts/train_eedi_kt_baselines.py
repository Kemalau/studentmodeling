#!/usr/bin/env python3
"""Train KT baselines on Eedi next-correctness C1 rows.

The input rows are produced by prepare_eedi_c1.py. Each row contains a short
student history, a query item/skill, and a binary correctness target.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def read_jsonl(path: Path, source_filter: str | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                source = str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "")
                if row.get("task") in {"next_correctness", "predict_correctness"} and (
                    source_filter is None or source == source_filter
                ):
                    rows.append(row)
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target(row: dict[str, Any]) -> int:
    return 1 if row["target_label"] == "correct" else 0


def meta(row: dict[str, Any], name: str) -> str:
    metadata = row.get("metadata") or {}
    value = str(metadata.get(name) or "")
    if name in {"skill_id", "question_id"}:
        source = str(metadata.get("mix_source") or row.get("dataset") or "unknown")
        return f"{source}::{value}"
    return value


def event_id(row: dict[str, Any], event: dict[str, Any], name: str) -> str:
    source = str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")
    if name == "question_id":
        value = str(event.get("question_id") or event.get("surface_id") or "")
    else:
        values = event.get("kc_ids") or []
        value = "|".join(str(item) for item in values) or str(event.get("skill_id") or event.get("family") or "")
    return f"{source}::{value}"


def auc_score(y_true: list[int], y_prob: list[float]) -> float | None:
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0])
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


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    scores = []
    for label in (0, 1):
        tp = sum(g == label and p == label for g, p in zip(y_true, y_pred))
        fp = sum(g != label and p == label for g, p in zip(y_true, y_pred))
        fn = sum(g == label and p != label for g, p in zip(y_true, y_pred))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def ece_score(y_true: list[int], y_prob: list[float], bins: int = 10) -> float:
    buckets = [[] for _ in range(bins)]
    for y, p_correct in zip(y_true, y_prob):
        pred = 1 if p_correct >= 0.5 else 0
        conf = p_correct if pred == 1 else 1 - p_correct
        idx = min(bins - 1, int(conf * bins))
        buckets[idx].append((1.0 if pred == y else 0.0, conf))
    out = 0.0
    total = len(y_true)
    for bucket in buckets:
        if not bucket:
            continue
        acc = sum(x for x, _ in bucket) / len(bucket)
        conf = sum(c for _, c in bucket) / len(bucket)
        out += len(bucket) / max(1, total) * abs(acc - conf)
    return out


def metrics(rows: list[dict[str, Any]], probs: list[float]) -> dict[str, Any]:
    y = [target(row) for row in rows]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    nll = [
        -math.log(max(min(p, 1 - 1e-7), 1e-7)) if gold == 1 else -math.log(max(min(1 - p, 1 - 1e-7), 1e-7))
        for gold, p in zip(y, probs)
    ]
    return {
        "n": len(rows),
        "accuracy": sum(int(a == b) for a, b in zip(y, preds)) / max(1, len(y)),
        "macro_f1": macro_f1(y, preds),
        "auc_positive_correct": auc_score(y, probs),
        "nll": sum(nll) / max(1, len(nll)),
        "brier": sum((p - gold) ** 2 for gold, p in zip(y, probs)) / max(1, len(y)),
        "ece": ece_score(y, probs),
        "target_counts": dict(Counter("correct" if v else "incorrect" for v in y)),
        "prediction_counts": dict(Counter("correct" if v else "incorrect" for v in preds)),
    }


def count_rates(rows: list[dict[str, Any]], field: str) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = meta(row, field)
        counts[key][0] += target(row)
        counts[key][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def laplace_rate(counts: dict[str, tuple[int, int]], key: str, global_rate: float) -> float:
    if key not in counts:
        return global_rate
    c, n = counts[key]
    return (c + 1.0) / (n + 2.0)


def build_feature_stats(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_rate = sum(target(row) for row in train_rows) / max(1, len(train_rows))
    return {
        "global_rate": global_rate,
        "skill_counts": count_rates(train_rows, "skill_id"),
        "question_counts": count_rates(train_rows, "question_id"),
    }


def row_features(row: dict[str, Any], stats: dict[str, Any]) -> list[float]:
    history = row.get("history") or []
    hist_y = [int(h.get("correctness", h.get("is_correct", 0))) for h in history]
    skill = meta(row, "skill_id")
    question = meta(row, "question_id")
    same_skill = [
        int(h.get("correctness", h.get("is_correct", 0)))
        for h in history
        if event_id(row, h, "skill_id") == skill
    ]
    last3 = hist_y[-3:]
    global_rate = float(stats["global_rate"])
    skill_rate = laplace_rate(stats["skill_counts"], skill, global_rate)
    question_rate = laplace_rate(stats["question_counts"], question, global_rate)
    correct_response = str((row.get("query") or {}).get("correct_response") or "")
    response_onehot = [1.0 if correct_response == x else 0.0 for x in ["A", "B", "C", "D"]]
    hist_len = len(hist_y)
    same_len = len(same_skill)
    return [
        1.0,
        global_rate,
        skill_rate,
        question_rate,
        (sum(hist_y) + 1.0) / (hist_len + 2.0),
        (sum(same_skill) + 1.0) / (same_len + 2.0),
        (sum(last3) + 1.0) / (len(last3) + 2.0),
        float(hist_y[-1]) if hist_y else global_rate,
        min(hist_len, 12) / 12.0,
        min(same_len, 12) / 12.0,
        *response_onehot,
    ]


class FeatureDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], stats: dict[str, Any]):
        self.rows = rows
        self.x = torch.tensor([row_features(row, stats) for row in rows], dtype=torch.float32)
        self.y = torch.tensor([target(row) for row in rows], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LogisticBaseline(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def all_ids(rows: list[dict[str, Any]], field: str) -> list[str]:
    values = set()
    for row in rows:
        values.add(meta(row, field))
        for h in row.get("history") or []:
            values.add(event_id(row, h, field))
    return sorted(x for x in values if x)


def build_maps(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "skill": {v: i + 1 for i, v in enumerate(all_ids(rows, "skill_id"))},
        "question": {v: i + 1 for i, v in enumerate(all_ids(rows, "question_id"))},
    }


def interaction_id(value: str, mapping: dict[str, int], correct: int) -> int:
    idx = mapping.get(value, 0)
    if idx == 0:
        return 1
    return 2 + (idx - 1) * 2 + int(correct)


class SequenceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        maps: dict[str, dict[str, int]],
        stats: dict[str, Any],
        max_len: int,
        interaction: str,
    ):
        self.rows = rows
        self.maps = maps
        self.stats = stats
        self.max_len = max_len
        self.interaction = interaction
        self.features = torch.tensor([row_features(row, stats) for row in rows], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        history = (row.get("history") or [])[-self.max_len :]
        ids: list[int] = []
        for h in history:
            name = event_id(row, h, "question_id" if self.interaction == "question" else "skill_id")
            mapping = self.maps["question" if self.interaction == "question" else "skill"]
            ids.append(interaction_id(name, mapping, int(h.get("correctness", 0))))
        length = max(1, len(ids))
        pad = self.max_len - len(ids)
        ids = [0] * pad + ids
        return {
            "seq": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "query_skill": torch.tensor(self.maps["skill"].get(meta(row, "skill_id"), 0), dtype=torch.long),
            "query_question": torch.tensor(self.maps["question"].get(meta(row, "question_id"), 0), dtype=torch.long),
            "features": self.features[idx],
            "y": torch.tensor(target(row), dtype=torch.float32),
        }


def collate_dict(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


class DKTModel(nn.Module):
    def __init__(
        self,
        n_interactions: int,
        n_skills: int,
        n_questions: int,
        feature_dim: int,
        variant: str,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.variant = variant
        self.emb = nn.Embedding(n_interactions, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        if variant == "skill_output":
            self.out = nn.Linear(hidden_dim, n_skills + 1)
        elif variant == "item_mlp":
            self.skill_emb = nn.Embedding(n_skills + 1, emb_dim, padding_idx=0)
            self.question_emb = nn.Embedding(n_questions + 1, emb_dim, padding_idx=0)
            self.out = nn.Sequential(
                nn.Linear(hidden_dim + emb_dim * 2 + feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(variant)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.emb(batch["seq"])
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            batch["length"].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h, _) = self.lstm(packed)
        state = self.dropout(h[-1])
        if self.variant == "skill_output":
            logits = self.out(state)
            return logits.gather(1, batch["query_skill"].clamp_min(0).unsqueeze(1)).squeeze(1)
        skill = self.skill_emb(batch["query_skill"])
        question = self.question_emb(batch["query_question"])
        return self.out(torch.cat([state, skill, question, batch["features"]], dim=1)).squeeze(1)


def predict_feature_model(model: nn.Module, rows: list[dict[str, Any]], stats: dict[str, Any], device: torch.device) -> list[float]:
    ds = FeatureDataset(rows, stats)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.extend(torch.sigmoid(model(x.to(device))).detach().cpu().tolist())
    return [float(p) for p in probs]


def predict_sequence_model(
    model: nn.Module,
    rows: list[dict[str, Any]],
    maps: dict[str, dict[str, int]],
    stats: dict[str, Any],
    max_len: int,
    interaction: str,
    device: torch.device,
) -> list[float]:
    ds = SequenceDataset(rows, maps, stats, max_len, interaction)
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_dict)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            probs.extend(torch.sigmoid(model(batch)).detach().cpu().tolist())
    return [float(p) for p in probs]


def train_logistic(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    train_ds = FeatureDataset(train_rows, stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = LogisticBaseline(train_ds.x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_valid = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
        valid_probs = predict_feature_model(model, valid_rows, stats, device)
        valid_metrics = metrics(valid_rows, valid_probs)
        history.append({"epoch": epoch, "train_loss": total / len(train_ds), "valid": valid_metrics})
        if valid_metrics["nll"] < best_valid:
            best_valid = valid_metrics["nll"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "history": history}


def train_dkt(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    maps: dict[str, dict[str, int]],
    stats: dict[str, Any],
    variant: str,
    interaction: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    max_len = args.max_seq_len
    train_ds = SequenceDataset(train_rows, maps, stats, max_len, interaction)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dict)
    n_interactions = (len(maps["question" if interaction == "question" else "skill"]) * 2) + 2
    feature_dim = len(row_features(train_rows[0], stats))
    model = DKTModel(
        n_interactions=n_interactions,
        n_skills=len(maps["skill"]),
        n_questions=len(maps["question"]),
        feature_dim=feature_dim,
        variant=variant,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_valid = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch["y"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(batch["y"])
        valid_probs = predict_sequence_model(model, valid_rows, maps, stats, max_len, interaction, device)
        valid_metrics = metrics(valid_rows, valid_probs)
        history.append({"epoch": epoch, "train_loss": total / len(train_ds), "valid": valid_metrics})
        if valid_metrics["nll"] < best_valid:
            best_valid = valid_metrics["nll"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "history": history}


def constant_probs(rows: list[dict[str, Any]], p: float) -> list[float]:
    return [float(p) for _ in rows]


def train_count_baselines(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = build_feature_stats(train_rows)
    global_rate = stats["global_rate"]
    skill_counts = stats["skill_counts"]
    question_counts = stats["question_counts"]
    out = {}
    out["global_rate"] = metrics(eval_rows, constant_probs(eval_rows, global_rate))
    out["skill_rate"] = metrics(
        eval_rows,
        [laplace_rate(skill_counts, meta(row, "skill_id"), global_rate) for row in eval_rows],
    )
    out["question_rate"] = metrics(
        eval_rows,
        [laplace_rate(question_counts, meta(row, "question_id"), global_rate) for row in eval_rows],
    )
    out["history_rate"] = metrics(
        eval_rows,
        [
            (sum(int(h.get("correctness", h.get("is_correct", 0))) for h in row.get("history", [])) + 1.0)
            / (len(row.get("history", [])) + 2.0)
            for row in eval_rows
        ],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=12)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-filter")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_rows = read_jsonl(args.train_jsonl, args.source_filter)
    valid_rows = read_jsonl(args.valid_jsonl, args.source_filter)
    test_rows = read_jsonl(args.test_jsonl, args.source_filter)
    stats = build_feature_stats(train_rows)
    maps = build_maps(train_rows + valid_rows)

    result: dict[str, Any] = {
        "train_jsonl": str(args.train_jsonl),
        "valid_jsonl": str(args.valid_jsonl),
        "test_jsonl": str(args.test_jsonl),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data": {
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "test_rows": len(test_rows),
            "n_skills": len(maps["skill"]),
            "n_questions": len(maps["question"]),
            "train_correct_rate": stats["global_rate"],
        },
        "baselines": {},
    }

    result["baselines"]["count_global_skill_question_history"] = train_count_baselines(train_rows, test_rows)

    logistic, logistic_info = train_logistic(train_rows, valid_rows, stats, args, device)
    result["baselines"]["logistic_features"] = {
        "test": metrics(test_rows, predict_feature_model(logistic, test_rows, stats, device)),
        "training": logistic_info,
    }

    dkt_skill, dkt_skill_info = train_dkt(
        train_rows, valid_rows, maps, stats, "skill_output", "skill", args, device
    )
    result["baselines"]["dkt_skill_lstm"] = {
        "test": metrics(
            test_rows,
            predict_sequence_model(dkt_skill, test_rows, maps, stats, args.max_seq_len, "skill", device),
        ),
        "training": dkt_skill_info,
    }

    dkt_item, dkt_item_info = train_dkt(
        train_rows, valid_rows, maps, stats, "item_mlp", "question", args, device
    )
    result["baselines"]["dkt_item_lstm_mlp"] = {
        "test": metrics(
            test_rows,
            predict_sequence_model(dkt_item, test_rows, maps, stats, args.max_seq_len, "question", device),
        ),
        "training": dkt_item_info,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    write_json(args.output_json, result)


if __name__ == "__main__":
    main()
