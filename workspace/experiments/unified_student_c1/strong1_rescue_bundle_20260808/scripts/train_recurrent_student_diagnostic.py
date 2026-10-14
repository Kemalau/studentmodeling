#!/usr/bin/env python3
"""Train a minimal recurrent student-state diagnostic on one GPU.

This is an M2 gate, not the final full run.  It initializes from the causal
one-shot CF1 adapter, updates state from one/four observed interactions, decodes
without raw history, and distills the recurrent state toward the full-history
one-shot state while replaying the direct task objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from recurrent_student_model import RecurrentStudentModel
from train_c1_lora import build_bottleneck_attention_mask, find_bottleneck_positions


STATE_TOKEN = "<STUDENT_STATE>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrajectoryDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def alias_text(text: str, alias: str) -> str:
    return text.replace(STATE_TOKEN, alias)


def encode_prompt_target(tokenizer, prompt: str, target: str, max_length: int) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    if len(prompt_ids) + len(target_ids) + len(eos) > max_length:
        raise ValueError(
            f"Sequence length {len(prompt_ids) + len(target_ids) + len(eos)} exceeds {max_length}"
        )
    input_ids = prompt_ids + target_ids + eos
    labels = [-100] * len(prompt_ids) + target_ids + eos
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "prompt_ends": torch.tensor([len(prompt_ids)], dtype=torch.long),
    }


def encode_prompt(tokenizer, prompt: str, max_length: int) -> dict[str, torch.Tensor]:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > max_length:
        raise ValueError(f"Prompt length {len(ids)} exceeds {max_length}")
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
    }


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def direct_supervised_batch(
    tokenizer, raw_prompt: str, target: str, alias: str, max_length: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Encode a one-shot view with the same strict late-fusion route."""
    encoded = encode_prompt_target(tokenizer, alias_text(raw_prompt, alias), target, max_length)
    history_start, state_pos = find_bottleneck_positions(tokenizer, raw_prompt, alias)
    attention_2d = encoded["attention_mask"]
    encoded["attention_mask"] = build_bottleneck_attention_mask(
        attention_2d,
        torch.tensor([history_start], dtype=torch.long),
        torch.tensor([state_pos], dtype=torch.long),
        encoded["prompt_ends"],
        style="late_fusion",
    )
    # For a repeated carrier alias, the generic bottleneck mask treats the
    # final carrier as the boundary and masks earlier carrier slots together
    # with history.  Restore all carrier slots only for the answer anchor and
    # answer rows; query rows remain unable to read any state slot.
    state_width = len(tokenizer(alias, add_special_tokens=False)["input_ids"])
    if state_width > 1:
        state_start = state_pos - state_width + 1
        if state_start < history_start:
            raise ValueError("Repeated carrier alias overlaps the history boundary")
        answer_anchor = int(encoded["prompt_ends"][0].item()) - 1
        encoded["attention_mask"][0, 0, answer_anchor:, state_start : state_pos + 1] = 0.0
    return move(encoded, device)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def full_history_state(
    backbone,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    state_token_id: int,
) -> torch.Tensor:
    positions = []
    for row in input_ids:
        match = torch.nonzero(row.eq(state_token_id), as_tuple=False).flatten()
        if match.numel() != 1:
            raise ValueError(f"Direct prompt has {match.numel()} state carriers, expected one")
        positions.append(match[0])
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, torch.stack(positions).to(hidden.device)]


def deterministic_full_history_state(
    backbone,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    state_token_id: int,
) -> torch.Tensor:
    """Read an online teacher state without stochastic LoRA dropout."""
    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            return full_history_state(
                backbone, input_ids, attention_mask, state_token_id
            ).detach()
    finally:
        backbone.train(was_training)


def state_statistics(states: list[torch.Tensor]) -> dict[str, float]:
    if not states:
        return {"state_variance": float("nan"), "mean_pairwise_cosine": float("nan")}
    matrix = torch.cat([state.detach().float().cpu() for state in states], dim=0)
    # Scalar states are [N,D]; layerwise trajectories are [N,L,D].  Compare
    # complete per-example states without collapsing or averaging away depth.
    matrix = matrix.reshape(matrix.shape[0], -1)
    variance = matrix.var(dim=0, unbiased=False).mean().item()
    normalized = F.normalize(matrix, dim=-1)
    similarities = normalized @ normalized.T
    if matrix.shape[0] > 1:
        mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
        mean_cosine = similarities[mask].mean().item()
    else:
        mean_cosine = 1.0
    return {"state_variance": variance, "mean_pairwise_cosine": mean_cosine}


def sequence_logprob(
    logits: torch.Tensor, input_ids: torch.Tensor, prompt_end: int
) -> float:
    score = 0.0
    for position in range(prompt_end, input_ids.shape[1]):
        log_probs = torch.log_softmax(logits[0, position - 1].float(), dim=-1)
        score += float(log_probs[input_ids[0, position]].item())
    return score


def gradient_audit(wrapper: RecurrentStudentModel) -> dict[str, Any]:
    groups = {
        "lora": [],
        "state_projection": [],
        "layerwise_transition": [],
        "other_trainable": [],
    }
    frozen_with_grad = []
    for name, parameter in wrapper.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                frozen_with_grad.append(name)
            continue
        if "lora_" in name:
            group = "lora"
        elif "state_in_proj" in name or "state_out_proj" in name:
            group = "state_projection"
        elif "transition_in" in name or "transition_out" in name:
            group = "layerwise_transition"
        else:
            group = "other_trainable"
        grad_sum = 0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum())
        groups[group].append({"name": name, "gradient_abs_sum": grad_sum})
    summary = {
        group: {
            "parameter_tensors": len(records),
            "tensors_with_nonzero_gradient": sum(
                record["gradient_abs_sum"] > 0.0 for record in records
            ),
            "gradient_abs_sum": sum(record["gradient_abs_sum"] for record in records),
        }
        for group, records in groups.items()
    }
    summary["frozen_tensors_with_grad"] = frozen_with_grad
    required_groups = (
        "lora",
        "state_projection",
        "layerwise_transition",
    )
    present = [group for group in required_groups if groups[group]]
    if not present:
        raise RuntimeError("Gradient audit failed: no supported trainable group")
    for group in present:
        if summary[group]["gradient_abs_sum"] <= 0.0:
            raise RuntimeError(
                f"Gradient audit failed: {group} received no gradient"
            )
    if frozen_with_grad:
        raise RuntimeError(f"Gradient audit failed: frozen tensors have gradients: {frozen_with_grad[:3]}")
    return summary


@torch.no_grad()
def evaluate(
    wrapper,
    tokenizer,
    rows: list[dict[str, Any]],
    alias: str,
    max_length: int,
    limit: int,
) -> dict[str, Any]:
    wrapper.eval()
    recurrent_losses = []
    direct_losses = []
    cosines = []
    states = []
    direct_states = []
    recurrent_correct = 0
    direct_correct = 0
    candidate_n = 0
    task_records: dict[str, dict[str, int]] = {}
    device = next(wrapper.parameters()).device
    for row in rows[:limit]:
        state = wrapper.initial_state(1, device=device)
        for update_text in row["update_texts"]:
            update = move(encode_prompt(tokenizer, alias_text(update_text, alias), max_length), device)
            state = wrapper.update_state(state, **update).state
        decode = move(
            encode_prompt_target(
                tokenizer, alias_text(row["decode_text"], alias), row["target"], max_length
            ),
            device,
        )
        prompt_ends = decode.pop("prompt_ends")
        output = wrapper.decode_from_state(state, **decode, prompt_ends=prompt_ends)
        recurrent_losses.append(causal_lm_loss(output.logits, decode["labels"]).item())
        direct_gold = direct_supervised_batch(
            tokenizer, row["direct_text"], row["target"], alias, max_length, device
        )
        direct_gold_prompt_ends = direct_gold.pop("prompt_ends")
        if hasattr(wrapper, "capture_direct_state"):
            captured = wrapper.capture_direct_state(
                input_ids=direct_gold["input_ids"],
                attention_mask=direct_gold["attention_mask"],
                labels=direct_gold["labels"],
            )
            teacher = captured.state.detach()
            direct_output = captured.model_output
        else:
            direct = move(
                encode_prompt(
                    tokenizer,
                    alias_text(row["direct_text"], alias),
                    max_length,
                ),
                device,
            )
            teacher = deterministic_full_history_state(
                wrapper.backbone,
                **direct,
                state_token_id=wrapper.state_token_id,
            )
            direct_output = wrapper.backbone(
                input_ids=direct_gold["input_ids"],
                attention_mask=direct_gold["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        if state is not None:
            cosines.append(
                F.cosine_similarity(
                    state.float(), teacher.float(), dim=-1
                ).mean().item()
            )
            states.append(state)
            direct_states.append(teacher)
        direct_losses.append(causal_lm_loss(direct_output.logits, direct_gold["labels"]).item())

        labels = list(row.get("candidate_labels") or [])
        targets = dict(row.get("candidate_targets") or {})
        if labels and all(str(label) in targets for label in labels):
            recurrent_scores: dict[str, float] = {}
            direct_scores: dict[str, float] = {}
            for label_value in labels:
                label = str(label_value)
                target = str(targets[label])
                candidate = move(
                    encode_prompt_target(
                        tokenizer, alias_text(row["decode_text"], alias), target, max_length
                    ),
                    device,
                )
                candidate_prompt_ends = candidate.pop("prompt_ends")
                candidate_output = wrapper.decode_from_state(
                    state, **candidate, prompt_ends=candidate_prompt_ends
                )
                recurrent_scores[label] = sequence_logprob(
                    candidate_output.logits,
                    candidate["input_ids"],
                    int(candidate_prompt_ends.item()),
                )

                direct_candidate = direct_supervised_batch(
                    tokenizer, row["direct_text"], target, alias, max_length, device
                )
                direct_candidate_prompt_ends = direct_candidate.pop("prompt_ends")
                direct_candidate["attention_mask"] = direct_candidate[
                    "attention_mask"
                ].to(dtype=wrapper.backbone.get_input_embeddings().weight.dtype)
                direct_candidate_output = wrapper.backbone(
                    input_ids=direct_candidate["input_ids"],
                    attention_mask=direct_candidate["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                direct_scores[label] = sequence_logprob(
                    direct_candidate_output.logits,
                    direct_candidate["input_ids"],
                    int(direct_candidate_prompt_ends.item()),
                )
            gold = str(row.get("target_label"))
            recurrent_prediction = max(recurrent_scores, key=recurrent_scores.get)
            direct_prediction = max(direct_scores, key=direct_scores.get)
            recurrent_correct += int(recurrent_prediction == gold)
            direct_correct += int(direct_prediction == gold)
            candidate_n += 1
            task = str(row.get("task"))
            task_record = task_records.setdefault(
                task, {"n": 0, "recurrent_correct": 0, "direct_correct": 0}
            )
            task_record["n"] += 1
            task_record["recurrent_correct"] += int(recurrent_prediction == gold)
            task_record["direct_correct"] += int(direct_prediction == gold)
    stats = state_statistics(states)
    direct_stats = state_statistics(direct_states)
    by_task = {
        task: {
            "n": record["n"],
            "recurrent_accuracy": record["recurrent_correct"] / record["n"],
            "direct_accuracy": record["direct_correct"] / record["n"],
            "accuracy_gap_recurrent_minus_direct": (
                record["recurrent_correct"] - record["direct_correct"]
            ) / record["n"],
        }
        for task, record in sorted(task_records.items())
    }
    return {
        "recurrent_gold_nll": sum(recurrent_losses) / max(len(recurrent_losses), 1),
        "direct_gold_nll": sum(direct_losses) / max(len(direct_losses), 1),
        "gold_nll_gap_recurrent_minus_direct": (
            sum(recurrent_losses) / max(len(recurrent_losses), 1)
            - sum(direct_losses) / max(len(direct_losses), 1)
        ),
        "candidate_n": candidate_n,
        "recurrent_accuracy": recurrent_correct / max(candidate_n, 1),
        "direct_accuracy": direct_correct / max(candidate_n, 1),
        "accuracy_gap_recurrent_minus_direct": (
            recurrent_correct - direct_correct
        ) / max(candidate_n, 1),
        "recurrent_direct_cosine": sum(cosines) / max(len(cosines), 1),
        "state_comparison_n": len(cosines),
        "cold_start_n": len(recurrent_losses) - len(cosines),
        "n": len(recurrent_losses),
        "by_task": by_task,
        "direct_state_variance": direct_stats["state_variance"],
        "direct_mean_pairwise_cosine": direct_stats["mean_pairwise_cosine"],
        **stats,
    }


def save_checkpoint(
    wrapper: RecurrentStudentModel,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    config: dict,
    history: list[dict[str, float]],
    order: list[int],
) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    adapter_dir = checkpoint / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    wrapper.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    checkpoint_payload = {
        "step": step,
        "state_in_proj": wrapper.state_in_proj.state_dict(),
        "state_out_proj": wrapper.state_out_proj.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "order": order,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "config": config,
    }
    temporary = checkpoint / "recurrent_state.pt.partial"
    torch.save(checkpoint_payload, temporary)
    temporary.replace(checkpoint / "recurrent_state.pt")
    receipt = {
        "status": "complete",
        "step": step,
        "sha256": {
            "adapter_model": sha256(adapter_dir / "adapter_model.safetensors"),
            "recurrent_state": sha256(checkpoint / "recurrent_state.pt"),
        },
    }
    receipt_temporary = checkpoint / "checkpoint_completion.json.partial"
    receipt_temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_temporary.replace(checkpoint / "checkpoint_completion.json")


def validate_resume_checkpoint(state_path: Path) -> dict[str, Any]:
    checkpoint = state_path.parent
    receipt_path = checkpoint / "checkpoint_completion.json"
    adapter_path = checkpoint / "adapter/adapter_model.safetensors"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Resume receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise ValueError(f"Resume checkpoint is not complete: {receipt}")
    expected = receipt.get("sha256") or {}
    actual = {
        "adapter_model": sha256(adapter_path),
        "recurrent_state": sha256(state_path),
    }
    if actual != expected:
        raise ValueError(
            f"Resume checkpoint hash mismatch: expected={expected} actual={actual}"
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--direct-replay-weight", type=float, default=0.25)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--trainable-projections", action="store_true")
    parser.add_argument("--state-projection-rank", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires one CUDA GPU")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    alias_ids = tokenizer(args.state_token_alias, add_special_tokens=False)["input_ids"]
    if len(alias_ids) != 1:
        raise ValueError(f"State alias must be one token, got {alias_ids}")
    state_token_id = int(alias_ids[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    adapter_source = Path(args.init_adapter)
    if args.resume_from_checkpoint:
        validate_resume_checkpoint(args.resume_from_checkpoint)
        resume_adapter = args.resume_from_checkpoint.parent / "adapter"
        if not (resume_adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Resume adapter is missing: {resume_adapter}")
        adapter_source = resume_adapter
    backbone = PeftModel.from_pretrained(base, str(adapter_source), is_trainable=True)
    wrapper = RecurrentStudentModel(
        backbone,
        state_token_id=state_token_id,
        trainable_projections=args.trainable_projections,
        state_projection_rank=args.state_projection_rank,
    )
    # The 4-bit backbone was already dispatched by ``device_map="auto"`` and
    # must not be moved again with ``Module.to``.  Move only the lightweight
    # recurrent modules that were created on CPU after dispatch.
    wrapper.state_norm.to(device)
    wrapper.state_in_proj.to(device)
    wrapper.state_out_proj.to(device)

    train_rows = read_jsonl(args.train_jsonl)
    valid_rows = read_jsonl(args.valid_jsonl)
    parameters = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)
    history = []
    gradient_receipt: dict[str, Any] | None = None
    existing_gradient_receipt = args.output_dir / "gradient_audit.json"
    if args.resume_from_checkpoint and existing_gradient_receipt.is_file():
        gradient_receipt = json.loads(existing_gradient_receipt.read_text(encoding="utf-8"))
    config = vars(args).copy()
    config.update(
        {
            "state_token_id": state_token_id,
            "train_rows": len(train_rows),
            "effective_init_adapter": str(adapter_source),
            "direct_view": "last_u_natural_history_strict_late_fusion",
            "decoder_view": "state_plus_current_query_strict_late_fusion",
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in wrapper.parameters() if parameter.requires_grad
            ),
            "state_projection_parameter_count": sum(
                parameter.numel()
                for name, parameter in wrapper.named_parameters()
                if parameter.requires_grad
                and ("state_in_proj" in name or "state_out_proj" in name)
            ),
            "micro_steps": args.max_steps,
            "planned_optimizer_steps": math.ceil(
                args.max_steps / args.gradient_accumulation_steps
            ),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    order = list(range(len(train_rows)))
    random.shuffle(order)
    start_step = 1
    if args.resume_from_checkpoint:
        payload = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
        wrapper.state_in_proj.load_state_dict(payload["state_in_proj"])
        wrapper.state_out_proj.load_state_dict(payload["state_out_proj"])
        optimizer.load_state_dict(payload["optimizer"])
        history = list(payload.get("history") or [])
        order = list(payload["order"])
        random.setstate(payload["python_random_state"])
        torch.set_rng_state(payload["torch_random_state"])
        if torch.cuda.is_available() and payload.get("cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])
        start_step = int(payload["step"]) + 1
        print(f"[resume] checkpoint={args.resume_from_checkpoint} next_step={start_step}", flush=True)
    stop_step = 0 if args.eval_only else args.max_steps
    for step in range(start_step, stop_step + 1):
        row = train_rows[order[(step - 1) % len(order)]]
        state = wrapper.initial_state(1, device=device)
        for update_text in row["update_texts"]:
            update = move(encode_prompt(tokenizer, alias_text(update_text, args.state_token_alias), args.max_length), device)
            state = wrapper.update_state(state, **update).state

        decode = move(
            encode_prompt_target(
                tokenizer,
                alias_text(row["decode_text"], args.state_token_alias),
                row["target"],
                args.max_length,
            ),
            device,
        )
        prompt_ends = decode.pop("prompt_ends")
        decoded = wrapper.decode_from_state(state, **decode, prompt_ends=prompt_ends)
        answer_loss = causal_lm_loss(decoded.logits, decode["labels"])

        direct_prompt = move(
            encode_prompt(tokenizer, alias_text(row["direct_text"], args.state_token_alias), args.max_length),
            device,
        )
        teacher_state = deterministic_full_history_state(
            wrapper.backbone, **direct_prompt, state_token_id=state_token_id
        )
        consistency_loss = 1.0 - F.cosine_similarity(
            state.float(), teacher_state.float(), dim=-1
        ).mean()

        direct_supervised = direct_supervised_batch(
            tokenizer,
            row["direct_text"],
            row["target"],
            args.state_token_alias,
            args.max_length,
            device,
        )
        direct_supervised.pop("prompt_ends")
        direct_output = wrapper.backbone(
            input_ids=direct_supervised["input_ids"],
            attention_mask=direct_supervised["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        direct_loss = causal_lm_loss(direct_output.logits, direct_supervised["labels"])
        loss = (
            answer_loss
            + args.consistency_weight * consistency_loss
            + args.direct_replay_weight * direct_loss
        ) / args.gradient_accumulation_steps
        loss.backward()
        if gradient_receipt is None:
            gradient_receipt = gradient_audit(wrapper)
            (args.output_dir / "gradient_audit.json").write_text(
                json.dumps(gradient_receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"gradient_audit": gradient_receipt}), flush=True)
        if step % args.gradient_accumulation_steps == 0 or step == args.max_steps:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "micro_step": step,
            "optimizer_step": math.ceil(step / args.gradient_accumulation_steps),
            "loss": float(loss.detach()) * args.gradient_accumulation_steps,
            "answer_loss": float(answer_loss.detach()),
            "consistency_loss": float(consistency_loss.detach()),
            "direct_loss": float(direct_loss.detach()),
        }
        history.append(record)
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
        if step % args.save_steps == 0:
            save_checkpoint(wrapper, optimizer, args.output_dir, step, config, history, order)

    metrics = evaluate(
        wrapper,
        tokenizer,
        valid_rows,
        args.state_token_alias,
        args.max_length,
        args.eval_limit,
    )
    print(json.dumps({"validation": metrics}, indent=2), flush=True)
    if args.eval_only:
        (args.output_dir / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "run_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return

    final_dir = args.output_dir / "final_adapter"
    wrapper.backbone.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    final_state_path = args.output_dir / "recurrent_state.pt"
    final_state_temporary = args.output_dir / "recurrent_state.pt.partial"
    torch.save(
        {
            "state_in_proj": wrapper.state_in_proj.state_dict(),
            "state_out_proj": wrapper.state_out_proj.state_dict(),
            "config": config,
        },
        final_state_temporary,
    )
    final_state_temporary.replace(final_state_path)
    (args.output_dir / "trainer_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    completion = {
        "status": "complete",
        "micro_steps": args.max_steps,
        "optimizer_steps": math.ceil(args.max_steps / args.gradient_accumulation_steps),
        "sha256": {
            "adapter_model": sha256(final_dir / "adapter_model.safetensors"),
            "recurrent_state": sha256(final_state_path),
            "validation_metrics": sha256(args.output_dir / "validation_metrics.json"),
            "run_config": sha256(args.output_dir / "run_config.json"),
            "gradient_audit": sha256(args.output_dir / "gradient_audit.json"),
        },
    }
    completion_temporary = args.output_dir / "completion.json.partial"
    completion_temporary.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completion_temporary.replace(args.output_dir / "completion.json")


if __name__ == "__main__":
    main()
