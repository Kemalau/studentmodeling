#!/usr/bin/env python3
"""Train the M3 unified recurrent student state with cached teacher KD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from recurrent_distillation import topk_teacher_cross_entropy
from recurrent_student_model import RecurrentStudentModel
from teacher_cache import TeacherCache
from train_recurrent_student_diagnostic import (
    alias_text,
    causal_lm_loss,
    deterministic_full_history_state,
    direct_supervised_batch,
    encode_prompt,
    encode_prompt_target,
    evaluate,
    gradient_audit,
    move,
    read_jsonl,
    sha256,
    validate_resume_checkpoint,
)


def cache_source_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_training_history(history: list[dict]) -> dict:
    """Report every task/source loss without pretending open tasks have accuracy."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in history:
        groups.setdefault((record["task"], record["source"]), []).append(record)
    fields = (
        "loss",
        "generation_loss",
        "behavior_kd_loss",
        "consistency_loss",
        "direct_loss",
    )
    return {
        f"{task}/{source}": {
            "n": len(records),
            **{
                f"mean_{field}": sum(float(record[field]) for record in records)
                / len(records)
                for field in fields
            },
        }
        for (task, source), records in sorted(groups.items())
    }


def save_checkpoint(
    wrapper: RecurrentStudentModel,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    config: dict,
    history: list[dict],
    order: list[int],
) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    adapter_dir = checkpoint / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    wrapper.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    payload = {
        "step": step,
        "state_in_proj": wrapper.state_in_proj.state_dict(),
        "state_out_proj": wrapper.state_out_proj.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "order": order,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all(),
        "config": config,
    }
    temporary = checkpoint / "recurrent_state.pt.partial"
    torch.save(payload, temporary)
    temporary.replace(checkpoint / "recurrent_state.pt")
    receipt = {
        "status": "complete",
        "step": step,
        "sha256": {
            "adapter_model": sha256(adapter_dir / "adapter_model.safetensors"),
            "recurrent_state": sha256(checkpoint / "recurrent_state.pt"),
        },
    }
    temporary_receipt = checkpoint / "checkpoint_completion.json.partial"
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary_receipt.replace(checkpoint / "checkpoint_completion.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--init-recurrent-state", type=Path)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--generation-weight", type=float, default=1.0)
    parser.add_argument("--behavior-kd-weight", type=float, default=0.0)
    parser.add_argument("--behavior-kd-temperature", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--direct-replay-weight", type=float, default=0.25)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--trainable-projections", action="store_true")
    parser.add_argument("--state-projection-rank", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights = (
        args.generation_weight,
        args.behavior_kd_weight,
        args.consistency_weight,
        args.direct_replay_weight,
    )
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Loss weights must be non-negative and not all zero")
    if args.behavior_kd_weight > 0 and args.teacher_cache is None:
        raise ValueError("Positive KD weight requires --teacher-cache")
    if args.behavior_kd_temperature <= 0:
        raise ValueError("KD temperature must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("M3 training requires CUDA")
    device = torch.device("cuda")
    train_rows = read_jsonl(args.train_jsonl)
    valid_rows = read_jsonl(args.valid_jsonl)
    cache = None
    if args.teacher_cache is not None:
        cache = TeacherCache(
            args.teacher_cache,
            expected_source_sha256=cache_source_sha(args.train_jsonl),
        )
        missing = [
            str(row["example_id"])
            for row in train_rows
            if str(row["example_id"]) not in cache.by_example
        ]
        if args.behavior_kd_weight > 0 and missing:
            raise ValueError(f"Teacher cache misses {len(missing)} train examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    alias_ids = tokenizer(args.state_token_alias, add_special_tokens=False)["input_ids"]
    if len(alias_ids) != 1:
        raise ValueError(f"State alias must be one token, got {alias_ids}")
    state_token_id = int(alias_ids[0])
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
        adapter_source = args.resume_from_checkpoint.parent / "adapter"
    backbone = PeftModel.from_pretrained(base, str(adapter_source), is_trainable=True)
    wrapper = RecurrentStudentModel(
        backbone,
        state_token_id=state_token_id,
        trainable_projections=args.trainable_projections,
        state_projection_rank=args.state_projection_rank,
    )
    wrapper.state_norm.to(device)
    wrapper.state_in_proj.to(device)
    wrapper.state_out_proj.to(device)
    if args.init_recurrent_state:
        initial_state_payload = torch.load(
            args.init_recurrent_state, map_location="cpu", weights_only=False
        )
        if "state_in_proj" not in initial_state_payload or "state_out_proj" not in initial_state_payload:
            raise ValueError("Initial recurrent state lacks projection weights")
        wrapper.state_in_proj.load_state_dict(initial_state_payload["state_in_proj"])
        wrapper.state_out_proj.load_state_dict(initial_state_payload["state_out_proj"])
    parameters = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)

    config = vars(args).copy()
    config.update(
        {
            "state_token_id": state_token_id,
            "train_rows": len(train_rows),
            "train_sha256": cache_source_sha(args.train_jsonl),
            "valid_sha256": cache_source_sha(args.valid_jsonl),
            "effective_init_adapter": str(adapter_source),
            "effective_init_recurrent_state": (
                str(args.init_recurrent_state) if args.init_recurrent_state else None
            ),
            "decoder_view": "state_plus_current_query_strict_late_fusion",
            "teacher_view": "cached_full_context_target_topk_distribution",
            "planned_optimizer_steps": math.ceil(
                args.max_steps / args.gradient_accumulation_steps
            ),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order = list(range(len(train_rows)))
    random.shuffle(order)
    history: list[dict] = []
    start_step = 1
    gradient_receipt = None
    if args.resume_from_checkpoint:
        payload = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
        wrapper.state_in_proj.load_state_dict(payload["state_in_proj"])
        wrapper.state_out_proj.load_state_dict(payload["state_out_proj"])
        optimizer.load_state_dict(payload["optimizer"])
        history = list(payload.get("history") or [])
        order = list(payload["order"])
        random.setstate(payload["python_random_state"])
        torch.set_rng_state(payload["torch_random_state"])
        torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])
        start_step = int(payload["step"]) + 1
        receipt_path = args.output_dir / "gradient_audit.json"
        if receipt_path.is_file():
            gradient_receipt = json.loads(receipt_path.read_text())

    wrapper.train()
    for step in range(start_step, args.max_steps + 1):
        row = train_rows[order[(step - 1) % len(order)]]
        state = wrapper.initial_state(1, device=device)
        for update_text in row["update_texts"]:
            update = move(
                encode_prompt(
                    tokenizer,
                    alias_text(update_text, args.state_token_alias),
                    args.max_length,
                ),
                device,
            )
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
        generation_loss = causal_lm_loss(decoded.logits, decode["labels"])

        direct_prompt = move(
            encode_prompt(
                tokenizer,
                alias_text(row["direct_text"], args.state_token_alias),
                args.max_length,
            ),
            device,
        )
        teacher_state = deterministic_full_history_state(
            wrapper.backbone, **direct_prompt, state_token_id=state_token_id
        )
        consistency_loss = 1.0 - F.cosine_similarity(
            state.float(), teacher_state.float(), dim=-1
        ).mean()

        direct = direct_supervised_batch(
            tokenizer,
            row["direct_text"],
            row["target"],
            args.state_token_alias,
            args.max_length,
            device,
        )
        direct.pop("prompt_ends")
        direct_output = wrapper.backbone(
            input_ids=direct["input_ids"],
            attention_mask=direct["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        direct_loss = causal_lm_loss(direct_output.logits, direct["labels"])

        kd_loss = decoded.logits.new_zeros((), dtype=torch.float32)
        if args.behavior_kd_weight > 0:
            cached = cache.load(str(row["example_id"]))
            kd_loss = topk_teacher_cross_entropy(
                decoded.logits,
                decode["labels"],
                torch.tensor(cached["target_token_ids"]),
                cached["teacher_topk_ids"],
                cached["teacher_topk_logprobs"],
                args.behavior_kd_temperature,
            )
        total = (
            args.generation_weight * generation_loss
            + args.behavior_kd_weight * kd_loss
            + args.consistency_weight * consistency_loss
            + args.direct_replay_weight * direct_loss
        )
        (total / args.gradient_accumulation_steps).backward()
        if gradient_receipt is None:
            gradient_receipt = gradient_audit(wrapper)
            (args.output_dir / "gradient_audit.json").write_text(
                json.dumps(gradient_receipt, indent=2, sort_keys=True) + "\n"
            )
        if step % args.gradient_accumulation_steps == 0 or step == args.max_steps:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "optimizer_step": math.ceil(step / args.gradient_accumulation_steps),
            "loss": float(total.detach()),
            "generation_loss": float(generation_loss.detach()),
            "behavior_kd_loss": float(kd_loss.detach()),
            "consistency_loss": float(consistency_loss.detach()),
            "direct_loss": float(direct_loss.detach()),
            "task": str(row["task"]),
            "source": str(row["source"]),
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
    final_dir = args.output_dir / "final_adapter"
    wrapper.backbone.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    recurrent_state = args.output_dir / "recurrent_state.pt"
    temporary_state = args.output_dir / "recurrent_state.pt.partial"
    torch.save(
        {
            "state_in_proj": wrapper.state_in_proj.state_dict(),
            "state_out_proj": wrapper.state_out_proj.state_dict(),
            "config": config,
        },
        temporary_state,
    )
    temporary_state.replace(recurrent_state)
    (args.output_dir / "trainer_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / "training_metrics_by_task_source.json").write_text(
        json.dumps(summarize_training_history(history), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n"
    )
    completion = {
        "status": "complete",
        "micro_steps": args.max_steps,
        "optimizer_steps": math.ceil(args.max_steps / args.gradient_accumulation_steps),
        "sha256": {
            "adapter_model": sha256(final_dir / "adapter_model.safetensors"),
            "recurrent_state": sha256(recurrent_state),
            "validation_metrics": sha256(args.output_dir / "validation_metrics.json"),
            "run_config": sha256(args.output_dir / "run_config.json"),
            "gradient_audit": sha256(args.output_dir / "gradient_audit.json"),
            "training_metrics_by_task_source": sha256(
                args.output_dir / "training_metrics_by_task_source.json"
            ),
        },
    }
    temporary_completion = args.output_dir / "completion.json.partial"
    temporary_completion.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    temporary_completion.replace(args.output_dir / "completion.json")
    print(json.dumps({"validation": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
