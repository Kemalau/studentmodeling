#!/usr/bin/env python3
"""Train response-aware neural specialists for public T2 and T4.

The model is deliberately a task specialist rather than another language
model.  A GRU encodes the observed question/response/correctness sequence.  A
query-conditioned candidate head predicts closed-choice responses, while a
correctness head predicts the T4 future outcome.  Open-response examples use
the same training-only response-mode / MalRule transfer fallbacks as the
frozen formal specialist protocol.  Final scores are always computed with the
canonical MathEq and JointMath evaluators.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from formal_metric_utils import extract_tag, generation_metrics


TASKS = {"predict_response", "predict_prospective_error"}
PUBLIC_LABEL = {"predict_response": "T2", "predict_prospective_error": "T4"}
PRIMARY_METRIC = {
    "predict_response": "math_equivalence",
    "predict_prospective_error": "joint_math_equivalence",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(path: Path, task: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("task")) != task:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_taskonly_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, valid = [], []
    for row in rows:
        original_split = str((row.get("metadata") or {}).get("original_split") or row.get("split") or "")
        (valid if original_split in {"valid", "validation", "val"} else train).append(row)
    if not train or not valid:
        # This path is useful for bounded smoke fixtures.  It is deterministic
        # by student and is never used by the formal task-only manifests, which
        # carry the original split explicitly.
        train, valid = [], []
        for row in rows:
            student = str(row.get("student_id") or row.get("example_id") or "")
            bucket = int(hashlib.sha256(student.encode("utf-8")).hexdigest()[:8], 16) % 10
            (valid if bucket == 0 else train).append(row)
    if not train or not valid:
        raise ValueError("both train and validation rows are required")
    return train, valid


def source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("eval_source") or metadata.get("mix_source") or row.get("dataset") or "unknown")


def event_outcome(event: dict[str, Any]) -> int:
    for key in ("correctness", "first_response_correctness", "is_correct"):
        if event.get(key) is not None:
            return int(event[key])
    return 0


def event_kcs(event: dict[str, Any]) -> set[str]:
    values = event.get("kc_ids") or []
    if values:
        return {str(value) for value in values}
    value = event.get("skill_id") or event.get("family")
    return {str(value)} if value else set()


def event_response(event: dict[str, Any]) -> str:
    for key in ("student_response_id", "student_response", "student_answer"):
        if event.get(key) not in (None, ""):
            return str(event[key])
    return ""


def event_correct_response(event: dict[str, Any]) -> str:
    for key in ("correct_response_id", "correct_response", "correct_answer"):
        if event.get(key) not in (None, ""):
            return str(event[key])
    return ""


def target_correctness(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") or {}
    if "target_correctness" in metadata:
        return int(metadata["target_correctness"])
    return int(str(extract_tag(str(row.get("target") or ""), "correctness") or "").casefold() == "correct")


def qid(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    query = row.get("query") or {}
    value = metadata.get("question_id") or query.get("question_id") or query.get("surface_id") or ""
    return f"{source(row)}::{value}"


def row_skill(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    value = metadata.get("skill_id") or metadata.get("family") or ""
    return f"{source(row)}::{value}"


class ResponseModeModel:
    """Training-only response fallback for examples without candidates."""

    def __init__(self, rows: list[dict[str, Any]], conditioned: bool):
        self.student_item: dict[tuple[str, str, int | None], Counter[str]] = defaultdict(Counter)
        self.item: dict[tuple[str, int | None], Counter[str]] = defaultdict(Counter)
        self.skill: dict[tuple[str, int | None], Counter[str]] = defaultdict(Counter)
        self.source: dict[tuple[str, int | None], Counter[str]] = defaultdict(Counter)
        for row in rows:
            label = str(row.get("target_label") or "")
            condition = target_correctness(row) if conditioned else None
            self.student_item[(str(row.get("student_id") or ""), qid(row), condition)][label] += 1
            self.item[(qid(row), condition)][label] += 1
            self.skill[(row_skill(row), condition)][label] += 1
            self.source[(source(row), condition)][label] += 1

    @staticmethod
    def mode(counter: Counter[str]) -> str | None:
        return max(counter, key=lambda value: (counter[value], value)) if counter else None

    def predict(self, row: dict[str, Any], condition: int | None) -> str:
        counters = [
            self.student_item.get((str(row.get("student_id") or ""), qid(row), condition), Counter()),
            self.item.get((qid(row), condition), Counter()),
            self.skill.get((row_skill(row), condition), Counter()),
            self.source.get((source(row), condition), Counter()),
        ]
        for counter in counters:
            value = self.mode(counter)
            if value is not None:
                return value
        labels = [str(value) for value in (row.get("candidate_labels") or [])]
        return labels[0] if labels else ""


def clean_numeric(value: str) -> Fraction | None:
    text = str(value or "").strip().replace("$$", "").replace("$", "").replace(",", "")
    text = text.replace("−", "-").replace("÷", "/")
    pattern = r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+\-]?(?:\d+(?:\.\d+)?|\.\d+))?"
    if not re.fullmatch(pattern, text):
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def fraction_text(value: Fraction) -> str:
    value = value.limit_denominator(10000)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def malrule_transfer(row: dict[str, Any]) -> str | None:
    query_correct = clean_numeric(str((row.get("query") or {}).get("correct_answer") or ""))
    pairs = []
    for event in row.get("history") or []:
        correct = clean_numeric(str(event.get("correct_answer") or ""))
        response = clean_numeric(str(event.get("student_answer") or ""))
        if correct is not None and response is not None:
            pairs.append((correct, response))
    if query_correct is None or not pairs:
        return None
    transforms: list[Any] = [lambda value: value]
    offset = median([response - correct for correct, response in pairs])
    transforms.append(lambda value, offset=offset: value + offset)
    ratios = [response / correct for correct, response in pairs if correct != 0]
    if ratios:
        ratio = median(ratios)
        transforms.append(lambda value, ratio=ratio: value * ratio)
    transforms.extend(
        [lambda value: -value, lambda value: 1 - value, lambda value: 1 / value if value != 0 else value]
    )
    best = min(
        enumerate(transforms),
        key=lambda item: (sum(abs(float(item[1](correct) - response)) for correct, response in pairs) / len(pairs), item[0]),
    )
    return fraction_text(best[1](query_correct))


def response_text(row: dict[str, Any], label: str) -> str:
    target = (row.get("candidate_targets") or {}).get(label)
    parsed = extract_tag(str(target or ""), "predicted_response")
    if parsed is not None:
        return parsed
    options = (row.get("query") or {}).get("options") or {}
    return str(options.get(label, label))


def grouped_generation(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["source"], record["protocol"], str(record.get("delta") or ""))].append(record)
    result = {}
    for (source_name, protocol, delta), group in sorted(groups.items()):
        name = "/".join(part for part in (source_name, protocol, f"delta={delta}" if delta else "") if part)
        result[name] = generation_metrics(group, task)
    result["overall"] = generation_metrics(records, task)
    return result


def namespaced(row: dict[str, Any], value: Any) -> str:
    return f"{source(row)}::{value}"


def question_id(row: dict[str, Any], event: dict[str, Any] | None = None) -> str:
    value = event if event is not None else (row.get("query") or {})
    raw = value.get("question_id") or value.get("surface_id") or ""
    return namespaced(row, raw)


def skill_id(row: dict[str, Any], event: dict[str, Any] | None = None) -> str:
    if event is None:
        query = row.get("query") or {}
        values = query.get("kc_ids") or []
        raw = "|".join(str(value) for value in values)
        if not raw:
            metadata = row.get("metadata") or {}
            raw = str(metadata.get("skill_id") or metadata.get("family") or "")
    else:
        values = event_kcs(event)
        raw = "|".join(sorted(values))
    return namespaced(row, raw)


class Vocab:
    def __init__(self, values: Iterable[str]):
        self.values = ["<PAD>", "<UNK>", *sorted(set(str(value) for value in values if str(value)))]
        self.index = {value: idx for idx, value in enumerate(self.values)}

    def get(self, value: str) -> int:
        return self.index.get(str(value), 1)

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class Vocabs:
    question: Vocab
    skill: Vocab
    response: Vocab

    @classmethod
    def build(cls, rows: list[dict[str, Any]]) -> "Vocabs":
        questions: list[str] = []
        skills: list[str] = []
        responses: list[str] = []
        for row in rows:
            questions.append(question_id(row))
            skills.append(skill_id(row))
            responses.extend(namespaced(row, label) for label in (row.get("candidate_labels") or []))
            responses.append(namespaced(row, row.get("target_label") or ""))
            for event in row.get("history") or []:
                questions.append(question_id(row, event))
                skills.append(skill_id(row, event))
                responses.append(namespaced(row, event_response(event)))
                responses.append(namespaced(row, event_correct_response(event)))
        return cls(Vocab(questions), Vocab(skills), Vocab(responses))

    def receipt(self) -> dict[str, int]:
        return {"questions": len(self.question), "skills": len(self.skill), "responses": len(self.response)}


def numeric_features(row: dict[str, Any]) -> list[float]:
    history = row.get("history") or []
    outcomes = [event_outcome(event) for event in history]
    query_kcs = set((row.get("query") or {}).get("kc_ids") or [])
    same = [
        event_outcome(event)
        for event in history
        if query_kcs.intersection(event_kcs(event))
    ]
    recent = outcomes[-3:]
    return [
        len(outcomes) / 32.0,
        sum(outcomes) / len(outcomes) if outcomes else 0.5,
        len(same) / 32.0,
        sum(same) / len(same) if same else 0.5,
        sum(recent) / len(recent) if recent else 0.5,
        float(outcomes[-1]) if outcomes else 0.5,
    ]


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], vocabs: Vocabs, max_history: int):
        self.rows = rows
        self.vocabs = vocabs
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        history = (row.get("history") or [])[-self.max_history :]
        events = [
            (
                self.vocabs.question.get(question_id(row, event)),
                self.vocabs.skill.get(skill_id(row, event)),
                self.vocabs.response.get(namespaced(row, event_response(event))),
                event_outcome(event),
            )
            for event in history
        ]
        labels = [str(value) for value in (row.get("candidate_labels") or [])]
        candidate_ids = [self.vocabs.response.get(namespaced(row, value)) for value in labels]
        target_label = str(row.get("target_label") or "")
        candidate_gold = labels.index(target_label) if target_label in labels else -1
        correct_label = str(
            (row.get("query") or {}).get("correct_response_id")
            or (row.get("query") or {}).get("correct_response")
            or (row.get("query") or {}).get("correct_answer")
            or ""
        )
        history_responses = [event_response(event) for event in history]
        candidate_aux = [
            [
                float(value == correct_label),
                history_responses.count(value) / max(1, len(history_responses)),
            ]
            for value in labels
        ]
        return {
            "row_index": index,
            "events": events,
            "query_question": self.vocabs.question.get(question_id(row)),
            "query_skill": self.vocabs.skill.get(skill_id(row)),
            "features": numeric_features(row),
            "candidate_ids": candidate_ids,
            "candidate_aux": candidate_aux,
            "candidate_gold": candidate_gold,
            "correctness": float(target_correctness(row)),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_history = max(1, max(len(item["events"]) for item in batch))
    max_candidates = max(1, max(len(item["candidate_ids"]) for item in batch))
    question = torch.zeros((batch_size, max_history), dtype=torch.long)
    skill = torch.zeros((batch_size, max_history), dtype=torch.long)
    response = torch.zeros((batch_size, max_history), dtype=torch.long)
    outcome = torch.zeros((batch_size, max_history), dtype=torch.long)
    lengths = torch.ones(batch_size, dtype=torch.long)
    candidate_ids = torch.zeros((batch_size, max_candidates), dtype=torch.long)
    candidate_aux = torch.zeros((batch_size, max_candidates, 2), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    for idx, item in enumerate(batch):
        events = item["events"] or [(0, 0, 0, 0)]
        lengths[idx] = len(events)
        for pos, values in enumerate(events):
            question[idx, pos], skill[idx, pos], response[idx, pos], outcome[idx, pos] = values
        count = len(item["candidate_ids"])
        if count:
            candidate_ids[idx, :count] = torch.tensor(item["candidate_ids"], dtype=torch.long)
            candidate_aux[idx, :count] = torch.tensor(item["candidate_aux"], dtype=torch.float32)
            candidate_mask[idx, :count] = True
    return {
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        "question": question,
        "skill": skill,
        "response": response,
        "outcome": outcome,
        "lengths": lengths,
        "query_question": torch.tensor([item["query_question"] for item in batch], dtype=torch.long),
        "query_skill": torch.tensor([item["query_skill"] for item in batch], dtype=torch.long),
        "features": torch.tensor([item["features"] for item in batch], dtype=torch.float32),
        "candidate_ids": candidate_ids,
        "candidate_aux": candidate_aux,
        "candidate_mask": candidate_mask,
        "candidate_gold": torch.tensor([item["candidate_gold"] for item in batch], dtype=torch.long),
        "correctness": torch.tensor([item["correctness"] for item in batch], dtype=torch.float32),
    }


class ResponseAwareGRU(nn.Module):
    def __init__(self, vocabs: Vocabs, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.question = nn.Embedding(len(vocabs.question), embedding_dim, padding_idx=0)
        self.skill = nn.Embedding(len(vocabs.skill), embedding_dim, padding_idx=0)
        self.response = nn.Embedding(len(vocabs.response), embedding_dim, padding_idx=0)
        self.outcome = nn.Embedding(2, embedding_dim // 2)
        event_dim = embedding_dim * 3 + embedding_dim // 2
        self.event_projection = nn.Sequential(nn.Linear(event_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.query_projection = nn.Sequential(nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU())
        context_dim = hidden_dim * 2 + 6
        self.dropout = nn.Dropout(dropout)
        self.correctness = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.candidate = nn.Sequential(
            nn.Linear(context_dim + embedding_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        events = torch.cat(
            [
                self.question(batch["question"]),
                self.skill(batch["skill"]),
                self.response(batch["response"]),
                self.outcome(batch["outcome"]),
            ],
            dim=-1,
        )
        events = self.event_projection(events)
        packed = nn.utils.rnn.pack_padded_sequence(
            events, batch["lengths"].detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.gru(packed)
        query = self.query_projection(
            torch.cat([self.question(batch["query_question"]), self.skill(batch["query_skill"])], dim=-1)
        )
        context = self.dropout(torch.cat([state[-1], query, batch["features"]], dim=-1))
        correctness_logits = self.correctness(context).squeeze(-1)
        candidates = self.response(batch["candidate_ids"])
        predicted_correctness = torch.sigmoid(correctness_logits).unsqueeze(1).unsqueeze(2)
        expanded = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        candidate_logits = self.candidate(
            torch.cat([expanded, candidates, batch["candidate_aux"], predicted_correctness.expand(-1, candidates.shape[1], -1)], dim=-1)
        ).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~batch["candidate_mask"], -1e9)
        return correctness_logits, candidate_logits


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def predict(
    model: ResponseAwareGRU,
    rows: list[dict[str, Any]],
    vocabs: Vocabs,
    max_history: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    loader = DataLoader(RowDataset(rows, vocabs, max_history), batch_size=batch_size, shuffle=False, collate_fn=collate)
    probabilities = [0.5] * len(rows)
    choices = [-1] * len(rows)
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move(raw_batch, device)
            correctness_logits, candidate_logits = model(batch)
            probs = torch.sigmoid(correctness_logits).detach().cpu().tolist()
            predicted = candidate_logits.argmax(dim=1).detach().cpu().tolist()
            for offset, row_index in enumerate(raw_batch["row_index"].tolist()):
                probabilities[row_index] = float(probs[offset])
                if raw_batch["candidate_mask"][offset].any():
                    choices[row_index] = int(predicted[offset])
    return probabilities, choices


def evaluate(
    task: str,
    model: ResponseAwareGRU,
    train_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    vocabs: Vocabs,
    max_history: int,
    batch_size: int,
    device: torch.device,
    include_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probabilities, choices = predict(model, rows, vocabs, max_history, batch_size, device)
    response_modes = ResponseModeModel(train_rows, conditioned=(task == "predict_prospective_error"))
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    for index, (row, probability, choice) in enumerate(zip(rows, probabilities, choices)):
        labels = [str(value) for value in (row.get("candidate_labels") or [])]
        predicted_correctness = int(probability >= 0.5)
        if 0 <= choice < len(labels):
            label = labels[choice]
            method = "response_aware_gru_candidate"
        elif task == "predict_response" and source(row) == "malrule":
            label = malrule_transfer(row)
            method = "systematic_error_transfer"
            if label is None:
                label = response_modes.predict(row, None)
                method = "training_response_mode"
        else:
            condition = predicted_correctness if task == "predict_prospective_error" else None
            label = response_modes.predict(row, condition)
            method = "training_response_mode"
        methods[method] += 1
        response = response_text(row, str(label or ""))
        metadata = row.get("metadata") or {}
        raw_output = f"<predicted_response>{response}</predicted_response>"
        if task == "predict_prospective_error":
            raw_output = (
                f"<correctness>{'correct' if predicted_correctness else 'incorrect'}</correctness>" + raw_output
            )
        records.append(
            {
                "source": source(row),
                "protocol": str(metadata.get("eval_protocol") or "unknown"),
                "delta": metadata.get("prospective_delta"),
                "target": str(row["target"]),
                "raw_output": raw_output,
            }
        )
        if include_predictions:
            predictions.append(
                {
                    "example_id": str(row.get("example_id") or ""),
                    "student_id": str(row.get("student_id") or ""),
                    "task": task,
                    "source": source(row),
                    "target": str(row["target"]),
                    "raw_output": raw_output,
                    "correctness_probability": probability,
                    "response_method": method,
                }
            )
    metrics = grouped_generation(records, task)
    return {"metrics": metrics, "prediction_methods": dict(methods)}, predictions


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.task not in TASKS:
        raise ValueError(args.task)
    set_seed(args.seed)
    all_train = read_rows(args.train_jsonl, args.task, args.max_train_rows)
    train_rows, valid_rows = split_taskonly_rows(all_train)
    test_rows = read_rows(args.test_jsonl, args.task, args.max_test_rows)
    if not test_rows:
        raise ValueError("test rows are empty")
    vocabs = Vocabs.build(train_rows + valid_rows)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ResponseAwareGRU(vocabs, args.embedding_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        RowDataset(train_rows, vocabs, args.max_history),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_score = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for raw_batch in loader:
            batch = move(raw_batch, device)
            correctness_logits, candidate_logits = model(batch)
            candidate_rows = batch["candidate_gold"] >= 0
            loss = torch.zeros((), device=device)
            if candidate_rows.any():
                loss = loss + nn.functional.cross_entropy(
                    candidate_logits[candidate_rows], batch["candidate_gold"][candidate_rows]
                )
            if args.task == "predict_prospective_error":
                loss = loss + args.correctness_loss_weight * nn.functional.binary_cross_entropy_with_logits(
                    correctness_logits, batch["correctness"]
                )
            if not loss.requires_grad:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch["correctness"])
            total_rows += len(batch["correctness"])
        validation, _ = evaluate(
            args.task,
            model,
            train_rows,
            valid_rows,
            vocabs,
            args.max_history,
            args.eval_batch_size,
            device,
            include_predictions=False,
        )
        primary = float(validation["metrics"]["overall"][PRIMARY_METRIC[args.task]])
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_rows), "validation_primary": primary}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if primary > best_score:
            best_score = primary
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    # The categorical response fallback has no tuned hyperparameters; as in
    # the frozen specialist it may consume train+validation only after model
    # selection.  Test rows remain untouched.
    final, predictions = evaluate(
        args.task,
        model,
        train_rows + valid_rows,
        test_rows,
        vocabs,
        args.max_history,
        args.eval_batch_size,
        device,
        include_predictions=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "vocab_sizes": vocabs.receipt()}, model_path)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "stage": "formal_neural_response_specialist_v1",
        "status": "complete",
        "task": args.task,
        "public_task": PUBLIC_LABEL[args.task],
        "model": "response-aware GRU sequence specialist",
        "selection_metric": PRIMARY_METRIC[args.task],
        "best_validation_primary": best_score,
        "training_history": history,
        "support": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "vocabulary": vocabs.receipt(),
        "metrics": final["metrics"],
        "prediction_methods": final["prediction_methods"],
        "protocol": {
            "candidate_scoring": "query-conditioned complete candidate set",
            "open_response_fallback": "training-only response mode; MalRule systematic transfer for T2",
            "primary_metric": PRIMARY_METRIC[args.task],
            "test_used_for_training_or_selection": False,
        },
        "artifacts": {
            "model_sha256": sha256(model_path),
            "predictions_sha256": sha256(predictions_path),
        },
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir"} and isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    write_json(args.output_dir / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--correctness-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--max-history", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    value = train(args)
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
