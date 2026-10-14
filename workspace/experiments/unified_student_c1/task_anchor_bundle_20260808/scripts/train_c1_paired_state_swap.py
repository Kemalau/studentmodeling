#!/usr/bin/env python3
"""QLoRA training with an explicit paired state-swap objective.

Each training item contains four sequences:

1. a normal curriculum example;
2. a donor-history + recipient-question counterfactual example;
3. the recipient-history + recipient-question factual example;
4. a matched counterfactual copy supervised with the donor answer.

For the treatment arm, the state-token residual stream of (2) is copied into
(4) after selected decoder blocks.  Sequence (3) is always retained with the
recipient's native answer, so the same recipient question is explicitly
supervised to produce B under its native state and A under the donor state.
The control arm uses an exact duplicate of (2) as sequence (4) and performs no
activation patch.  This keeps sequence count, targets, and loss weights matched
without giving the control arm contradictory labels for identical inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from paired_state_quartet import build_quartet, patch_state_residual
from train_c1_lora import (
    STATE_TOKEN,
    TARGET_MODULES,
    StateTokenTrainer,
    add_special_state_token,
    assert_bottleneck_mask,
    build_bottleneck_attention_mask,
    encode_example,
    read_jsonl,
    resize_and_mean_init_new_tokens,
)


TASK = "malrule_next_response"
ARMS = ("control", "paired_swap")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_decoder_layers(model):
    candidates = (
        lambda value: value.model.layers,
        lambda value: value.model.model.layers,
        lambda value: value.base_model.model.model.layers,
    )
    for getter in candidates:
        try:
            layers = getter(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise AttributeError("Could not find decoder layers")


def copy_token_rows(model, source_id: int, target_id: int) -> None:
    """Initialize a newly-added token exactly from an existing token row."""
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings().weight
        input_embeddings[target_id].copy_(input_embeddings[source_id])
        output_layer = model.get_output_embeddings()
        if output_layer is not None and output_layer.weight.shape[0] > target_id:
            output_layer.weight[target_id].copy_(output_layer.weight[source_id])


class PairedStateSwapDataset(Dataset):
    def __init__(
        self,
        base_jsonl: Path,
        normal_jsonl: Path,
        pair_manifest: Path,
        tokenizer,
        max_length: int,
        arm: str,
        max_pairs: int | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm={arm!r}")
        base_rows = read_jsonl(base_jsonl, task_filter=TASK)
        by_id = {str(row["example_id"]): row for row in base_rows}
        normal_rows = read_jsonl(normal_jsonl, task_filter=TASK)
        payload = json.loads(pair_manifest.read_text(encoding="utf-8"))
        if payload.get("task") != TASK:
            raise ValueError(f"Pair manifest task={payload.get('task')!r}, expected {TASK!r}")
        if payload.get("source_sha256") != sha256(base_jsonl):
            raise ValueError("Pair manifest source SHA256 does not match base JSONL")
        specs = list(payload.get("pairs") or [])
        if max_pairs is not None:
            specs = specs[:max_pairs]
        if not specs:
            raise ValueError("Pair manifest is empty")

        examples = []
        for pair_id, spec in enumerate(specs):
            donor = by_id[str(spec["donor_id"])]
            recipient = by_id[str(spec["recipient_id"])]
            donor_target = str(spec["donor_target_on_recipient_query"])
            native_target = str(recipient["target_label"])
            manifest_native_target = str(spec.get("recipient_target", native_target))
            if native_target != manifest_native_target:
                raise AssertionError(
                    f"Recipient target drift for pair {pair_id}: "
                    f"row={native_target!r} manifest={manifest_native_target!r}"
                )
            normal_row = normal_rows[pair_id % len(normal_rows)]
            rows, swap_pairs, targets = build_quartet(
                donor, recipient, normal_row, donor_target, pair_id, arm
            )
            encoded = [
                encode_example(
                    tokenizer,
                    row,
                    max_length,
                    supervise_eos=True,
                    bottleneck_mask=True,
                    state_token_alias=None,
                )
                for row in rows
            ]
            if arm == "paired_swap":
                native_prefix = encoded[2]["input_ids"][: int(encoded[2]["prompt_end"])]
                destination_prefix = encoded[3]["input_ids"][: int(encoded[3]["prompt_end"])]
                if native_prefix != destination_prefix:
                    raise AssertionError(
                        f"Native and swapped recipient prompts differ for pair {pair_id}"
                    )
                if encoded[2]["state_pos"] != encoded[3]["state_pos"]:
                    raise AssertionError(
                        f"Native and swapped state positions differ for pair {pair_id}"
                    )
            examples.append(
                {
                    "sequences": encoded,
                    "swap_pairs": swap_pairs,
                    "loss_weights": [1.0, 1.0, 1.0, 1.0],
                    "pair_id": pair_id,
                    "targets": targets,
                }
            )
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class PairedStateSwapCollator:
    def __init__(self, tokenizer, bottleneck_style: str = "late_fusion") -> None:
        self.tokenizer = tokenizer
        self.bottleneck_style = bottleneck_style
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sequences: list[dict[str, Any]] = []
        loss_weights: list[float] = []
        source_indices: list[int] = []
        destination_indices: list[int] = []
        for item in items:
            offset = len(sequences)
            sequences.extend(item["sequences"])
            loss_weights.extend(float(value) for value in item["loss_weights"])
            for source, destination in item["swap_pairs"]:
                source_indices.append(offset + int(source))
                destination_indices.append(offset + int(destination))

        max_len = max(len(sequence["input_ids"]) for sequence in sequences)
        input_ids, attention, labels = [], [], []
        history_starts, state_positions, prompt_ends = [], [], []
        for sequence in sequences:
            ids = list(sequence["input_ids"])
            labs = list(sequence["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)
            labels.append(labs + [-100] * pad)
            history_starts.append(int(sequence["history_start"]))
            state_positions.append(int(sequence["state_pos"]))
            prompt_ends.append(int(sequence["prompt_end"]))

        attention_2d = torch.tensor(attention, dtype=torch.long)
        state_tensor = torch.tensor(state_positions, dtype=torch.long)
        attention_4d = build_bottleneck_attention_mask(
            attention_2d,
            torch.tensor(history_starts, dtype=torch.long),
            state_tensor,
            torch.tensor(prompt_ends, dtype=torch.long),
            self.bottleneck_style,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": attention_4d,
            "labels": torch.tensor(labels, dtype=torch.long),
            "state_positions": state_tensor,
            "swap_source_indices": torch.tensor(source_indices, dtype=torch.long),
            "swap_destination_indices": torch.tensor(destination_indices, dtype=torch.long),
            "sequence_loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
        }


class PairedStateSwapTrainer(StateTokenTrainer):
    """Trainer with persistent hooks so patches survive checkpoint recomputation."""

    def __init__(self, *args, patch_layers: list[int] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._state_positions: torch.Tensor | None = None
        self._source_indices: torch.Tensor | None = None
        self._destination_indices: torch.Tensor | None = None
        self._hook_handles = []
        layers = get_decoder_layers(self.model)
        selected = list(range(len(layers))) if patch_layers is None else list(patch_layers)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(f"patch_layers must be non-empty and unique, got {selected}")
        if min(selected) < 0 or max(selected) >= len(layers):
            raise ValueError(f"patch_layers={selected} outside model depth={len(layers)}")
        self.patch_layers = selected
        for layer_index in selected:
            self._hook_handles.append(layers[layer_index].register_forward_hook(self._patch_state_hook))

    def _patch_state_hook(self, _module, _inputs, output):
        if self._source_indices is None or self._source_indices.numel() == 0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        patched = patch_state_residual(
            hidden,
            self._state_positions,
            self._source_indices,
            self._destination_indices,
        )
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        self._state_positions = inputs.pop("state_positions")
        self._source_indices = inputs.pop("swap_source_indices")
        self._destination_indices = inputs.pop("swap_destination_indices")
        weights = inputs.pop("sequence_loss_weights")
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits[:, :-1, :].contiguous().float()
        shifted_labels = labels[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shifted_labels)
        valid = shifted_labels.ne(-100)
        per_sequence = (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        weights = weights.to(per_sequence.device, dtype=per_sequence.dtype)
        loss = (per_sequence * weights).sum() / weights.sum().clamp_min(1e-12)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--normal-jsonl", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--valid-base-jsonl", type=Path)
    parser.add_argument("--valid-normal-jsonl", type=Path)
    parser.add_argument("--valid-pair-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-valid-pairs", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--state-token", default=STATE_TOKEN)
    parser.add_argument("--init-state-token-from", default="<|vision_pad|>")
    parser.add_argument("--bottleneck-style", choices=("standard", "late_fusion"), default="late_fusion")
    parser.add_argument(
        "--patch-layers",
        default="all",
        help="Comma-separated decoder layer indices or 'all'. The quartet fix is independent of this ablation.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=750)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_tokenizer_len = len(tokenizer)
    source_ids = tokenizer(args.init_state_token_from, add_special_tokens=False)["input_ids"]
    if len(source_ids) != 1:
        raise ValueError(f"Initialization token must be one token, got {source_ids}")
    added = add_special_state_token(tokenizer, args.state_token)
    state_ids = tokenizer(args.state_token, add_special_tokens=False)["input_ids"]
    if len(state_ids) != 1:
        raise ValueError(f"State token must be one token, got {state_ids}")
    source_id, state_id = int(source_ids[0]), int(state_ids[0])

    train_dataset = PairedStateSwapDataset(
        args.base_jsonl,
        args.normal_jsonl,
        args.pair_manifest,
        tokenizer,
        args.max_length,
        args.arm,
        args.max_pairs,
    )
    eval_dataset = None
    if args.valid_base_jsonl and args.valid_normal_jsonl and args.valid_pair_manifest:
        eval_dataset = PairedStateSwapDataset(
            args.valid_base_jsonl,
            args.valid_normal_jsonl,
            args.valid_pair_manifest,
            tokenizer,
            args.max_length,
            args.arm,
            args.max_valid_pairs,
        )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",
    )
    resize_and_mean_init_new_tokens(model, len(tokenizer), base_tokenizer_len)
    copy_token_rows(model, source_id, state_id)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
            trainable_token_indices=[state_id],
        ),
    )
    model.print_trainable_parameters()

    model_depth = len(get_decoder_layers(model))
    if args.patch_layers == "all":
        patch_layers = list(range(model_depth))
    else:
        patch_layers = [int(value) for value in args.patch_layers.split(",") if value.strip()]
        if not patch_layers:
            raise ValueError("--patch-layers resolved to an empty list")

    first = train_dataset[0]["sequences"][0]
    toy_attention = torch.tensor([first["attention_mask"]], dtype=torch.long)
    toy_mask = build_bottleneck_attention_mask(
        toy_attention,
        torch.tensor([int(first["history_start"])]),
        torch.tensor([int(first["state_pos"])]),
        torch.tensor([int(first["prompt_end"])]),
        args.bottleneck_style,
    )
    assert_bottleneck_mask(
        toy_mask,
        int(first["history_start"]),
        int(first["state_pos"]),
        len(first["input_ids"]),
        int(first["prompt_end"]),
        args.bottleneck_style,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=max(1, args.max_steps // 20),
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = PairedStateSwapTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PairedStateSwapCollator(tokenizer, args.bottleneck_style),
        processing_class=tokenizer,
        save_embedding_layers=False,
        patch_layers=patch_layers,
    )
    resume = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume)
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    run_config = {
        "arm": args.arm,
        "model": args.model,
        "base_jsonl": str(args.base_jsonl),
        "normal_jsonl": str(args.normal_jsonl),
        "pair_manifest": str(args.pair_manifest),
        "base_sha256": sha256(args.base_jsonl),
        "normal_sha256": sha256(args.normal_jsonl),
        "pair_manifest_sha256": sha256(args.pair_manifest),
        "state_token": args.state_token,
        "state_token_id": state_id,
        "state_token_added": added,
        "init_state_token_from": args.init_state_token_from,
        "init_state_token_from_id": source_id,
        "bottleneck_style": args.bottleneck_style,
        "pair_structure": "normal+donor_A+native_B+counterfactual_A_v2",
        "patch_layers": patch_layers,
        "train_pairs": len(train_dataset),
        "valid_pairs": len(eval_dataset) if eval_dataset is not None else 0,
        "max_steps": args.max_steps,
        "resume_from_checkpoint": resume,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
