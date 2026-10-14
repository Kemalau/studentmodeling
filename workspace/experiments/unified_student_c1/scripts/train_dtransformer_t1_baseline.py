#!/usr/bin/env python3
"""Train the authors' DTransformer on the frozen five-source T1 protocol.

The model implementation is imported from a pinned checkout of
https://github.com/yxonic/DTransformer.  This file owns only the adapter from
our JSONL benchmark to the official model and the frozen evaluation protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler


OFFICIAL_REPOSITORY = "https://github.com/yxonic/DTransformer.git"
OFFICIAL_COMMIT = "1dc459804911b3900ed23d9f6b1d7407d2e919a8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("mix_source") or row.get("dataset") or "unknown")


def event_skill(source: str, event: dict[str, Any]) -> str:
    values = event.get("kc_ids") or []
    value = "|".join(str(item) for item in values)
    if not value:
        value = str(event.get("skill_id") or event.get("family") or "unknown")
    return f"{source}::{value}"


def event_question(source: str, event: dict[str, Any]) -> str:
    value = event.get("question_id") or event.get("surface_id") or "unknown"
    return f"{source}::{value}"


@dataclass(frozen=True)
class Sample:
    source: str
    skills: tuple[str, ...]
    questions: tuple[str, ...]
    responses: tuple[int, ...]
    label: int


def sample_from_row(row: dict[str, Any], max_seq_len: int) -> Sample:
    source = row_source(row)
    metadata = row.get("metadata") or {}
    events = list(row.get("history") or [])[-(max_seq_len - 1) :]
    skills = [event_skill(source, event) for event in events]
    questions = [event_question(source, event) for event in events]
    responses = [int(event.get("correctness", event.get("is_correct", 0))) for event in events]
    skills.append(f"{source}::{metadata.get('skill_id') or 'unknown'}")
    questions.append(f"{source}::{metadata.get('question_id') or 'unknown'}")
    label = int(row["target_label"] == "correct")
    responses.append(label)
    return Sample(source, tuple(skills), tuple(questions), tuple(responses), label)


def read_samples(path: Path, max_seq_len: int, limit: int | None = None) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "predict_correctness":
                continue
            samples.append(sample_from_row(row, max_seq_len))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def vocabulary(samples: list[Sample], field: str) -> dict[str, int]:
    values: set[str] = set()
    for sample in samples:
        values.update(getattr(sample, field))
    # Zero is reserved for a genuinely unseen concept/item at evaluation time.
    return {value: index + 1 for index, value in enumerate(sorted(values))}


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[Sample],
        skill_map: dict[str, int],
        question_map: dict[str, int],
        max_seq_len: int,
    ) -> None:
        self.samples = samples
        self.skill_map = skill_map
        self.question_map = question_map
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        length = len(sample.responses)
        pad = self.max_seq_len - length
        if pad < 0:
            raise ValueError(f"sample length {length} exceeds max_seq_len={self.max_seq_len}")
        skills = [self.skill_map.get(value, 0) for value in sample.skills] + [-1] * pad
        questions = [self.question_map.get(value, 0) for value in sample.questions] + [-1] * pad
        responses = list(sample.responses) + [-1] * pad
        return {
            "skills": torch.tensor(skills, dtype=torch.long),
            "questions": torch.tensor(questions, dtype=torch.long),
            "responses": torch.tensor(responses, dtype=torch.long),
            "target_position": torch.tensor(length - 1, dtype=torch.long),
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


class ContrastiveBatchSampler(Sampler[list[int]]):
    """Keep length>=5 sequences together so DTransformer CL is not disabled.

    The authors' implementation skips contrastive learning for a whole batch
    whenever its shortest sequence has fewer than five interactions.  Random
    batching would therefore disable CL for most batches on this benchmark,
    despite almost all rows being long enough individually.
    """

    def __init__(self, samples: list[Sample], batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.long = [index for index, sample in enumerate(samples) if len(sample.responses) >= 5]
        self.short = [index for index, sample in enumerate(samples) if len(sample.responses) < 5]

    def __len__(self) -> int:
        return math.ceil(len(self.long) / self.batch_size) + math.ceil(len(self.short) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [list(self.long), list(self.short)]
        batches: list[list[int]] = []
        for indices in groups:
            generator.shuffle(indices)
            batches.extend(indices[start : start + self.batch_size] for start in range(0, len(indices), self.batch_size))
        generator.shuffle(batches)
        yield from batches


def import_official_model(repository: Path, expected_commit: str) -> tuple[type[torch.nn.Module], str]:
    model_file = repository / "DTransformer" / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"official DTransformer checkout not found: {model_file}")
    marker = repository / "PINNED_COMMIT"
    if not marker.is_file():
        raise FileNotFoundError(f"missing official-code provenance marker: {marker}")
    actual_commit = marker.read_text(encoding="utf-8").strip()
    if actual_commit != expected_commit:
        raise RuntimeError(f"DTransformer commit mismatch: expected {expected_commit}, got {actual_commit}")
    # Submission bundles exclude nested .git metadata.  When it is available
    # locally, require the checkout HEAD to agree with the immutable marker.
    if (repository / ".git").exists():
        checkout_commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        if checkout_commit != actual_commit:
            raise RuntimeError(f"DTransformer checkout/marker mismatch: {checkout_commit} != {actual_commit}")
    sys.path.insert(0, str(repository))
    from DTransformer.model import DTransformer  # type: ignore[import-not-found]

    return DTransformer, actual_commit


def selected_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    records: list[tuple[int, float]] = []
    with torch.inference_mode():
        for batch in loader:
            skills = batch["skills"].to(device, non_blocking=True)
            questions = batch["questions"].to(device, non_blocking=True)
            responses = batch["responses"].to(device, non_blocking=True)
            positions = batch["target_position"].to(device, non_blocking=True)
            # The frozen row protocol exposes the target response only as a
            # label.  Keep sequence length but never expose that response to
            # the model, even though DTransformer's final causal block should
            # mathematically mask the current interaction.
            responses = responses.clone()
            responses.scatter_(1, positions.unsqueeze(1), 0)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits, *_ = model.predict(skills, responses, questions)
                selected = logits.gather(1, positions.unsqueeze(1)).squeeze(1)
                probabilities = torch.sigmoid(selected).float().cpu().numpy()
            indices = batch["index"].numpy()
            records.extend((int(index), float(probability)) for index, probability in zip(indices, probabilities))
    records.sort()
    return (
        np.asarray([index for index, _ in records], dtype=np.int64),
        np.asarray([probability for _, probability in records], dtype=np.float64),
    )


def target_only_contrastive_loss(
    model: torch.nn.Module,
    skills: torch.Tensor,
    responses: torch.Tensor,
    questions: torch.Tensor,
    positions: torch.Tensor,
    labels: torch.Tensor,
    lambda_cl: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Authors' DTransformer representation loss plus protocol-fair target BCE.

    The upstream training helper applies predictive BCE to every position.
    Our benchmark defines only the final queried interaction as supervised, so
    the predictive term is restricted to that position.  The representation
    perturbation and InfoNCE computation follow the authors' implementation.
    """

    model_responses = responses.clone()
    model_responses.scatter_(1, positions.unsqueeze(1), 0)
    logits, z_1, _, reg_loss, _ = model.predict(skills, model_responses, questions)
    selected_logits = logits.gather(1, positions.unsqueeze(1)).squeeze(1)
    pred_loss = F.binary_cross_entropy_with_logits(selected_logits, labels)

    lengths = (model_responses >= 0).sum(dim=1)
    min_length = int(lengths.min().item())
    if min_length < 5 or lambda_cl == 0:
        cl_loss = pred_loss.new_zeros(())
        return pred_loss + reg_loss, pred_loss, cl_loss

    augmented_skills = skills.clone()
    augmented_responses = model_responses.clone()
    augmented_questions = questions.clone()
    for batch_index in range(len(skills)):
        length = int(lengths[batch_index].item())
        swap_count = max(1, int(length * model.dropout_rate))
        for index in random.sample(range(length - 1), swap_count):
            for tensor in (augmented_skills, augmented_responses, augmented_questions):
                left = tensor[batch_index, index].clone()
                tensor[batch_index, index] = tensor[batch_index, index + 1]
                tensor[batch_index, index + 1] = left
        flip_count = max(1, int(length * model.dropout_rate))
        for index in random.sample(range(length), flip_count):
            augmented_responses[batch_index, index] = 1 - augmented_responses[batch_index, index]

    _, z_2, *_ = model.predict(augmented_skills, augmented_responses, augmented_questions)
    similarities = model.sim(z_1[:, :min_length, :], z_2[:, :min_length, :])
    contrastive_targets = (
        torch.arange(len(skills), device=skills.device)[:, None].expand(-1, min_length)
    )
    cl_loss = F.cross_entropy(similarities, contrastive_targets)
    return pred_loss + lambda_cl * cl_loss + reg_loss, pred_loss, cl_loss


def metric_values(samples: list[Sample], probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    predictions = (probabilities >= 0.5).astype(np.int64)

    def compute(indices: list[int]) -> dict[str, Any]:
        y = labels[indices]
        p = probabilities[indices]
        pred = predictions[indices]
        return {
            "n": len(indices),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None,
            "nll": float(log_loss(y, p, labels=[0, 1])),
        }

    result = compute(list(range(len(samples))))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.source].append(index)
    result["by_source"] = {name: compute(indices) for name, indices in sorted(groups.items())}
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--official-commit", default=OFFICIAL_COMMIT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-fc", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-know", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda-cl", type=float, default=0.1)
    parser.add_argument("--no-cl", action="store_true")
    parser.add_argument("--proj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-neg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--limit-test", type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_bf16 = bool(args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    DTransformer, official_commit = import_official_model(args.official_repo, args.official_commit)

    train = read_samples(args.train_jsonl, args.max_seq_len, args.limit_train)
    valid = read_samples(args.valid_jsonl, args.max_seq_len, args.limit_valid)
    test = read_samples(args.test_jsonl, args.max_seq_len, args.limit_test)
    if not train or not valid or not test:
        raise ValueError("train, valid, and test must each contain T1 rows")
    skill_map = vocabulary(train, "skills")
    question_map = vocabulary(train, "questions")
    train_dataset = SequenceDataset(train, skill_map, question_map, args.max_seq_len)
    valid_dataset = SequenceDataset(valid, skill_map, question_map, args.max_seq_len)
    test_dataset = SequenceDataset(test, skill_map, question_map, args.max_seq_len)
    train_sampler = ContrastiveBatchSampler(train, args.batch_size, args.seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, pin_memory=device.type == "cuda")

    model = DTransformer(
        n_questions=len(skill_map),
        n_pid=len(question_map),
        d_model=args.d_model,
        d_fc=args.d_fc,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lambda_cl=args.lambda_cl,
        proj=args.proj,
        hard_neg=args.hard_neg,
        window=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_nll = float("inf")
    remaining_patience = args.patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            skills = batch["skills"].to(device, non_blocking=True)
            questions = batch["questions"].to(device, non_blocking=True)
            responses = batch["responses"].to(device, non_blocking=True)
            positions = batch["target_position"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss, _, _ = target_only_contrastive_loss(
                    model,
                    skills,
                    responses,
                    questions,
                    positions,
                    labels,
                    0.0 if args.no_cl else args.lambda_cl,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_rows = len(skills)
            total_loss += float(loss.detach()) * batch_rows
            total_rows += batch_rows

        _, valid_probabilities = selected_probabilities(model, valid_loader, device, use_bf16)
        valid_labels = np.asarray([sample.label for sample in valid], dtype=np.int64)
        valid_nll = float(log_loss(valid_labels, valid_probabilities, labels=[0, 1]))
        valid_auc = float(roc_auc_score(valid_labels, valid_probabilities))
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "valid_nll": valid_nll,
            "valid_auc": valid_auc,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if valid_nll < best_nll:
            best_nll = valid_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = args.patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    model.load_state_dict(best_state)
    test_indices, test_probabilities = selected_probabilities(model, test_loader, device, use_bf16)
    if not np.array_equal(test_indices, np.arange(len(test))):
        raise RuntimeError("test predictions are missing or out of order")
    result = {
        "architecture": "authors' official DTransformer",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": official_commit,
        "best_epoch": best_epoch,
        "data": {"train": len(train), "valid": len(valid), "test": len(test)},
        "artifact_sha256": {
            "train_jsonl": sha256(args.train_jsonl),
            "valid_jsonl": sha256(args.valid_jsonl),
            "test_jsonl": sha256(args.test_jsonl),
        },
        "n_skills": len(skill_map),
        "n_questions": len(question_map),
        "contrastive_batching": {
            "eligible_rows": len(train_sampler.long),
            "ineligible_rows": len(train_sampler.short),
            "minimum_length": 5,
        },
        "supervision_protocol": {
            "predictive_bce": "target_only",
            "target_response_input": "constant_zero",
            "history_outcomes": "input_only",
            "contrastive_objective": "authors_info_nce_representation_perturbation",
        },
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"train_jsonl", "valid_jsonl", "test_jsonl", "output_json", "official_repo"}
        },
        "history": history,
        "test": metric_values(test, test_probabilities),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
