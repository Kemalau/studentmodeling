#!/usr/bin/env python3
"""Train the official pyKT AKT architecture on canonical T1 prompt histories."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset


turtle_stub = types.ModuleType("turtle")
turtle_stub.forward = None
sys.modules.setdefault("turtle", turtle_stub)
from pykt.models.akt import AKT  # noqa: E402


def read(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("task") == "predict_correctness":
                    rows.append(row)
    return rows


def source(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")


def event_skill(row: dict[str, Any], event: dict[str, Any]) -> str:
    values = event.get("kc_ids") or []
    value = "|".join(str(item) for item in values) or str(event.get("skill_id") or event.get("family") or "unknown")
    return f"{source(row)}::{value}"


def event_question(row: dict[str, Any], event: dict[str, Any]) -> str:
    value = event.get("question_id") or event.get("surface_id") or "unknown"
    return f"{source(row)}::{value}"


def maps(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    skills, questions = set(), set()
    for row in rows:
        skills.add(f"{source(row)}::{(row.get('metadata') or {}).get('skill_id', 'unknown')}")
        questions.add(f"{source(row)}::{(row.get('metadata') or {}).get('question_id', 'unknown')}")
        for event in row.get("history") or []:
            skills.add(event_skill(row, event))
            questions.add(event_question(row, event))
    return (
        {value: index + 1 for index, value in enumerate(sorted(skills))},
        {value: index + 1 for index, value in enumerate(sorted(questions))},
    )


class T1SequenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], skill_map: dict[str, int], question_map: dict[str, int], max_len: int):
        self.rows = rows
        self.skill_map = skill_map
        self.question_map = question_map
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        events = list((row.get("history") or [])[-(self.max_len - 1):])
        skills = [self.skill_map.get(event_skill(row, event), 0) for event in events]
        questions = [self.question_map.get(event_question(row, event), 0) for event in events]
        responses = [int(event.get("correctness", event.get("is_correct", 0))) for event in events]
        skills.append(self.skill_map.get(f"{source(row)}::{(row.get('metadata') or {}).get('skill_id', 'unknown')}", 0))
        questions.append(self.question_map.get(f"{source(row)}::{(row.get('metadata') or {}).get('question_id', 'unknown')}", 0))
        responses.append(int(row["target_label"] == "correct"))
        target_position = len(skills) - 1
        pad = self.max_len - len(skills)
        skills.extend([0] * pad)
        questions.extend([0] * pad)
        responses.extend([0] * pad)
        return {
            "skills": torch.tensor(skills, dtype=torch.long),
            "questions": torch.tensor(questions, dtype=torch.long),
            "responses": torch.tensor(responses, dtype=torch.long),
            "target_position": torch.tensor(target_position, dtype=torch.long),
            "label": torch.tensor(int(row["target_label"] == "correct"), dtype=torch.float32),
        }


def predict(model: AKT, rows: list[dict[str, Any]], skill_map: dict[str, int], question_map: dict[str, int], args: argparse.Namespace, device: torch.device) -> np.ndarray:
    loader = DataLoader(T1SequenceDataset(rows, skill_map, question_map, args.max_seq_len), batch_size=args.batch_size)
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            skills = batch["skills"].to(device)
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            positions = batch["target_position"].to(device)
            outputs, _ = model(skills, responses, questions)
            selected = outputs.gather(1, positions.unsqueeze(1)).squeeze(1)
            probabilities.extend(selected.cpu().tolist())
    return np.asarray(probabilities, dtype=float)


def metric_values(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([int(row["target_label"] == "correct") for row in rows])
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "n": len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    result["by_source"] = {}
    for source_name in sorted({source(row) for row in rows}):
        indices = [index for index, row in enumerate(rows) if source(row) == source_name]
        source_labels = labels[indices]
        source_probabilities = probabilities[indices]
        source_predictions = predictions[indices]
        result["by_source"][source_name] = {
            "n": len(indices),
            "accuracy": float(accuracy_score(source_labels, source_predictions)),
            "macro_f1": float(f1_score(source_labels, source_predictions, average="macro", zero_division=0)),
            "auc": float(roc_auc_score(source_labels, source_probabilities)) if len(set(source_labels.tolist())) == 2 else None,
            "nll": float(log_loss(source_labels, source_probabilities, labels=[0, 1])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    train, valid, test = read(args.train_jsonl), read(args.valid_jsonl), read(args.test_jsonl)
    skill_map, question_map = maps(train)
    device = torch.device(args.device)
    model = AKT(
        n_question=len(skill_map) + 1,
        n_pid=len(question_map) + 1,
        d_model=64,
        n_blocks=2,
        dropout=0.2,
        d_ff=256,
        final_fc_dim=256,
        num_attn_heads=8,
        l2=1e-5,
    ).to(device)
    train_loader = DataLoader(
        T1SequenceDataset(train, skill_map, question_map, args.max_seq_len),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_state = copy.deepcopy(model.state_dict())
    best_nll = float("inf")
    best_epoch = 0
    patience = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            skills = batch["skills"].to(device)
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            positions = batch["target_position"].to(device)
            labels = batch["label"].to(device)
            outputs, regularization = model(skills, responses, questions)
            selected = outputs.gather(1, positions.unsqueeze(1)).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy(selected, labels) + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
        valid_probabilities = predict(model, valid, skill_map, question_map, args, device)
        valid_nll = log_loss([int(row["target_label"] == "correct") for row in valid], valid_probabilities, labels=[0, 1])
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "valid_nll": valid_nll})
        if valid_nll < best_nll:
            best_nll = valid_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                break
    model.load_state_dict(best_state)
    test_probabilities = predict(model, test, skill_map, question_map, args, device)
    result = {
        "architecture": "official pyKT AKT",
        "best_epoch": best_epoch,
        "data": {"train": len(train), "valid": len(valid), "test": len(test)},
        "n_skills": len(skill_map),
        "n_questions": len(question_map),
        "history": history,
        "test": metric_values(test, test_probabilities),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
