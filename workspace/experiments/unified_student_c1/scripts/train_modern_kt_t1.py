#!/usr/bin/env python3
"""Train modern pyKT models on the frozen five-source T1 protocol.

Each JSONL row is a short, already-frozen student history followed by one
target interaction.  The target response is used only as the BCE label; the
causal model input at that position contains the target item/KC and a dummy
response, preventing target leakage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset


PYKT_COMMIT = "3aabdc3cab06a57e745518b070d31bcbb823af2c"
MODEL_CHOICES = ("akt", "simplekt", "dtransformer", "rekt", "ukt")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") == "predict_correctness":
                rows.append(row)
    return rows


def source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("eval_source") or metadata.get("mix_source") or row.get("dataset") or "unknown")


def event_skill(row: dict[str, Any], event: dict[str, Any]) -> str:
    values = event.get("kc_ids") or []
    value = "|".join(str(item) for item in values) or str(event.get("skill_id") or event.get("family") or "unknown")
    return f"{source(row)}::{value}"


def event_question(row: dict[str, Any], event: dict[str, Any]) -> str:
    value = event.get("question_id") or event.get("surface_id") or "unknown"
    return f"{source(row)}::{value}"


def target_skill(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    query = row.get("query") or {}
    values = query.get("kc_ids") or []
    value = metadata.get("skill_id") or ("|".join(str(item) for item in values) if values else "unknown")
    return f"{source(row)}::{value}"


def target_question(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    query = row.get("query") or {}
    value = metadata.get("question_id") or query.get("question_id") or query.get("surface_id") or "unknown"
    return f"{source(row)}::{value}"


def maps(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    skills, questions = set(), set()
    for row in rows:
        skills.add(target_skill(row))
        questions.add(target_question(row))
        for event in row.get("history") or []:
            skills.add(event_skill(row, event))
            questions.add(event_question(row, event))
    # Zero is reserved for unseen IDs at validation/test time.
    return (
        {value: index + 1 for index, value in enumerate(sorted(skills))},
        {value: index + 1 for index, value in enumerate(sorted(questions))},
    )


class T1Dataset(Dataset):
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
        skills.append(self.skill_map.get(target_skill(row), 0))
        questions.append(self.question_map.get(target_question(row), 0))
        target_position = len(skills) - 1
        pad = self.max_len - len(skills)
        skills.extend([0] * pad)
        questions.extend([0] * pad)
        # The model receives a dummy target response, but padding responses are
        # -1 so DTransformer can infer true sequence lengths.
        responses.extend([0] + [-1] * pad)
        return {
            "skills": torch.tensor(skills, dtype=torch.long),
            "questions": torch.tensor(questions, dtype=torch.long),
            "responses": torch.tensor(responses, dtype=torch.long),
            "target_position": torch.tensor(target_position, dtype=torch.long),
            "label": torch.tensor(int(row["target_label"] == "correct"), dtype=torch.float32),
        }


def import_model_class(pykt_root: Path, model_name: str):
    pinned = (pykt_root / "PINNED_COMMIT").read_text(encoding="utf-8").strip()
    if pinned != PYKT_COMMIT:
        raise ValueError(f"pyKT commit drift: expected {PYKT_COMMIT}, got {pinned}")
    # pyKT imports turtle.forward even in headless environments.
    turtle_stub = types.ModuleType("turtle")
    turtle_stub.forward = None
    sys.modules.setdefault("turtle", turtle_stub)
    # Load only the requested official module. Importing pykt.models eagerly
    # imports every optional architecture (and their unrelated dependencies).
    package_root = pykt_root / "pykt"
    pykt_package = types.ModuleType("pykt")
    pykt_package.__path__ = [str(package_root)]
    models_package = types.ModuleType("pykt.models")
    models_package.__path__ = [str(package_root / "models")]
    sys.modules["pykt"] = pykt_package
    sys.modules["pykt.models"] = models_package
    module_name = f"pykt.models.{model_name}"
    spec = importlib.util.spec_from_file_location(module_name, package_root / "models" / f"{model_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    names = {
        "akt": "AKT",
        "simplekt": "simpleKT",
        "dtransformer": "DTransformer",
        "rekt": "ReKT",
        "ukt": "UKT",
    }
    return getattr(module, names[model_name])


def init_model(model_name: str, model_class, n_skills: int, n_questions: int, args: argparse.Namespace):
    if model_name == "akt":
        return model_class(
            n_question=n_skills,
            n_pid=n_questions,
            d_model=args.d_model,
            n_blocks=2,
            dropout=args.dropout,
            d_ff=256,
            final_fc_dim=256,
            num_attn_heads=8,
            l2=1e-5,
            emb_type="qid",
        )
    if model_name == "simplekt":
        return model_class(
            n_question=n_skills,
            n_pid=n_questions,
            d_model=args.d_model,
            n_blocks=2,
            dropout=args.dropout,
            d_ff=256,
            final_fc_dim=256,
            final_fc_dim2=128,
            num_attn_heads=8,
            seq_len=args.max_seq_len,
            emb_type="qid",
        )
    if model_name == "dtransformer":
        return model_class(
            n_question=n_skills,
            n_pid=n_questions,
            d_model=args.d_model,
            d_ff=256,
            num_attn_heads=8,
            n_know=16,
            n_blocks=3,
            dropout=args.dropout,
            lambda_cl=0.0,
            emb_type="qid",
        )
    if model_name == "rekt":
        return model_class(
            skill_max=n_skills,
            pro_max=n_questions,
            d=args.d_model,
            dropout=args.dropout,
            emb_type="qid",
        )
    if model_name == "ukt":
        # CL is disabled because the official release expects augmented-response
        # fields that are not part of this frozen row protocol.
        return model_class(
            n_question=n_skills,
            n_pid=n_questions,
            d_model=args.d_model,
            n_blocks=2,
            dropout=args.dropout,
            d_ff=256,
            final_fc_dim=256,
            final_fc_dim2=128,
            num_attn_heads=8,
            seq_len=args.max_seq_len,
            use_CL=False,
            use_uncertainty_aug=False,
            emb_type="stoc_qid",
        )
    raise AssertionError(model_name)


def model_probabilities(model_name: str, model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    skills = batch["skills"]
    questions = batch["questions"]
    responses = batch["responses"]
    positions = batch["target_position"]
    if model_name in {"akt", "simplekt", "ukt"}:
        clean_responses = responses.clamp_min(0)
        dcur = {
            "qseqs": questions[:, :1],
            "cseqs": skills[:, :1],
            "rseqs": clean_responses[:, :1],
            "shft_qseqs": questions[:, 1:],
            "shft_cseqs": skills[:, 1:],
            "shft_rseqs": clean_responses[:, 1:],
            "masks": torch.ones_like(skills[:, 1:], dtype=torch.bool),
        }
        if model_name == "akt":
            # AKT's zero-peek attention prevents the target response from
            # contributing to the selected target position. Passing a fixed
            # dummy response nevertheless makes this invariant explicit.
            outputs, _ = model(skills, clean_responses, questions)
        else:
            outputs = model(dcur)
    elif model_name == "dtransformer":
        # Official DTransformer represents padding with -1 only while deriving
        # lengths; embedding lookup itself requires a non-negative response.
        # Our fixed-width batches therefore pass clean responses and an explicit
        # per-row length to the unchanged official forward/readout blocks.
        clean_responses = responses.clamp_min(0)
        q_emb, s_emb, _, p_diff = model.embedding(skills, clean_responses, questions)
        lengths = positions + 1
        z, _, _ = model(q_emb, s_emb, lengths)
        h = model.readout(z, q_emb)
        logits = model.out(torch.cat([q_emb, h], dim=-1)).squeeze(-1)
        outputs = torch.sigmoid(logits)
    elif model_name == "rekt":
        clean_responses = responses.clamp_min(0)
        # ReKT predicts each next item, so prepend a harmless dummy interaction.
        dcur = {
            "qseqs": torch.cat([torch.zeros_like(questions[:, :1]), questions[:, :-1]], dim=1),
            "cseqs": torch.cat([torch.zeros_like(skills[:, :1]), skills[:, :-1]], dim=1),
            "rseqs": torch.cat([torch.zeros_like(clean_responses[:, :1]), clean_responses[:, :-1]], dim=1),
            "shft_qseqs": questions,
            "shft_cseqs": skills,
            "shft_rseqs": clean_responses,
        }
        outputs = model(dcur)
    else:
        raise AssertionError(model_name)
    return outputs.gather(1, positions.unsqueeze(1)).squeeze(1)


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def predict(model_name: str, model, loader: DataLoader, device: torch.device) -> np.ndarray:
    result = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            result.extend(model_probabilities(model_name, model, move(batch, device)).cpu().tolist())
    return np.asarray(result, dtype=float)


def metrics(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([int(row["target_label"] == "correct") for row in rows])
    predictions = (probabilities >= 0.5).astype(int)

    def one(indices: list[int]) -> dict[str, Any]:
        gold, prob, pred = labels[indices], probabilities[indices], predictions[indices]
        return {
            "n": len(indices),
            "accuracy": float(accuracy_score(gold, pred)),
            "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
            "auc": float(roc_auc_score(gold, prob)) if len(set(gold.tolist())) == 2 else None,
            "nll": float(log_loss(gold, prob, labels=[0, 1])),
        }

    output = one(list(range(len(rows))))
    output["by_source"] = {}
    for source_name in sorted({source(row) for row in rows}):
        output["by_source"][source_name] = one([index for index, row in enumerate(rows) if source(row) == source_name])
    real_temporal = {"dbe_kt22", "foundationalassist", "xes3g5m"}
    output["real_temporal"] = one([index for index, row in enumerate(rows) if source(row) in real_temporal])
    output["real_all"] = one([index for index, row in enumerate(rows) if source(row) != "malrule"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--pykt-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=13)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train, valid, test = read_rows(args.train_jsonl), read_rows(args.valid_jsonl), read_rows(args.test_jsonl)
    if args.max_train_samples:
        train = train[: args.max_train_samples]
    skill_map, question_map = maps(train)
    model_class = import_model_class(args.pykt_root, args.model)
    model = init_model(args.model, model_class, len(skill_map) + 1, len(question_map) + 1, args)
    device = torch.device(args.device)
    model = model.to(device)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        T1Dataset(train, skill_map, question_map, args.max_seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(T1Dataset(valid, skill_map, question_map, args.max_seq_len), batch_size=args.batch_size)
    test_loader = DataLoader(T1Dataset(test, skill_map, question_map, args.max_seq_len), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_state = copy.deepcopy(model.state_dict())
    best_nll = math.inf
    best_epoch = 0
    remaining = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            probability = model_probabilities(args.model, model, batch)
            loss = torch.nn.functional.binary_cross_entropy(probability, batch["label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["label"])
        valid_probabilities = predict(args.model, model, valid_loader, device)
        valid_nll = float(log_loss([int(row["target_label"] == "correct") for row in valid], valid_probabilities, labels=[0, 1]))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "valid_nll": valid_nll})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if valid_nll < best_nll:
            best_nll, best_epoch = valid_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining = args.patience
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    probabilities = predict(args.model, model, test_loader, device)
    result = {
        "stage": "modern_kt_t1_five_source_v1",
        "status": "complete",
        "architecture": args.model,
        "pykt_commit": PYKT_COMMIT,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "data": {"train": len(train), "valid": len(valid), "test": len(test)},
        "n_skills": len(skill_map),
        "n_questions": len(question_map),
        "config": {
            "d_model": args.d_model,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "selection_metric": "validation_nll",
        },
        "history": history,
        "test": metrics(test, probabilities),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.predictions_jsonl.open("w", encoding="utf-8") as handle:
        for row, probability in zip(test, probabilities):
            handle.write(json.dumps({
                "example_id": row["example_id"],
                "source": source(row),
                "gold": row["target_label"],
                "p_correct": float(probability),
                "prediction": "correct" if probability >= 0.5 else "incorrect",
            }, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
