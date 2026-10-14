#!/usr/bin/env python3
"""LoRA SFT for C1 unified student modeling samples."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import peft.utils.save_and_load as peft_save_and_load
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
STATE_TOKEN = "<STUDENT_STATE>"
DEFAULT_STATE_TOKEN_ALIAS = "<|vision_pad|>"
HISTORY_OPEN = "<student_history>"
BOTTLENECK_STYLES = ("standard", "late_fusion")


@contextmanager
def non_tensor_parallel_peft_resume(model: torch.nn.Module) -> Iterator[None]:
    """Bypass PEFT's TP sharding hook only for a replicated single-device model.

    PEFT 0.19.1 imports tensor-parallel classes before it checks whether any
    layer is actually tensor parallel. Transformers 4.57.6 no longer exports
    ``EmbeddingParallel``, so ordinary single-GPU Trainer checkpoint resume
    otherwise fails during adapter reload. Preserve the native PEFT path when
    a layer has an active TP plan and device mesh, and fail closed if the
    compatibility hook cannot be restored exactly.
    """
    tensor_parallel_layers = [
        name
        for name, module in model.named_modules()
        if getattr(module, "_hf_tp_plan", None) is not None
        and getattr(module, "_hf_device_mesh", None) is not None
    ]
    if tensor_parallel_layers:
        raise RuntimeError(
            "The non-TP PEFT resume compatibility path cannot load a real "
            f"tensor-parallel model: {tensor_parallel_layers[:5]}"
        )
    original = peft_save_and_load._maybe_shard_state_dict_for_tp
    peft_save_and_load._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None
    try:
        yield
    finally:
        peft_save_and_load._maybe_shard_state_dict_for_tp = original
        if peft_save_and_load._maybe_shard_state_dict_for_tp is not original:
            raise RuntimeError("Failed to restore PEFT tensor-parallel resume hook")


def add_special_state_token(tokenizer, state_token: str) -> bool:
    before = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": [state_token]})
    ids = tokenizer(state_token, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{state_token} must tokenize to one id after registration, got {ids}")
    return len(tokenizer) > before


def encode_prompt_text(prompt: str, state_token_alias: str | None) -> str:
    if not state_token_alias:
        return prompt
    if STATE_TOKEN not in prompt:
        raise ValueError(f"Could not find {STATE_TOKEN} in prompt")
    return prompt.replace(STATE_TOKEN, state_token_alias)


def resize_and_mean_init_new_tokens(model, tokenizer_len: int, old_vocab_size: int) -> None:
    if tokenizer_len <= old_vocab_size:
        return
    model.resize_token_embeddings(tokenizer_len)
    input_emb = model.get_input_embeddings()
    with torch.no_grad():
        old_weight = input_emb.weight[:old_vocab_size].detach()
        mean_row = old_weight.mean(dim=0, keepdim=True).to(input_emb.weight.dtype)
        input_emb.weight[old_vocab_size:tokenizer_len].copy_(mean_row)
        output_emb = model.get_output_embeddings()
        if output_emb is not None and output_emb.weight.shape[0] >= tokenizer_len:
            old_out = output_emb.weight[:old_vocab_size].detach()
            mean_out = old_out.mean(dim=0, keepdim=True).to(output_emb.weight.dtype)
            output_emb.weight[old_vocab_size:tokenizer_len].copy_(mean_out)


def read_jsonl(
    path: Path,
    limit: int | None = None,
    task_filter: str | None = None,
    exclude_task: str | None = None,
) -> list[dict[str, Any]]:
    if task_filter and exclude_task:
        raise ValueError("Use either task_filter or exclude_task, not both")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if task_filter and row.get("task") != task_filter:
                    continue
                if exclude_task and row.get("task") == exclude_task:
                    continue
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def find_bottleneck_positions(tokenizer, prompt: str, state_token_alias: str | None = None) -> tuple[int, int]:
    state_char = prompt.find(STATE_TOKEN)
    if state_char < 0:
        raise ValueError(f"Could not find {STATE_TOKEN} in prompt")
    history_char = prompt.find(HISTORY_OPEN)
    if history_char < 0:
        raise ValueError(f"Could not find {HISTORY_OPEN} in prompt")
    history_start = len(tokenizer(prompt[:history_char], add_special_tokens=False)["input_ids"])
    through_state = prompt[:state_char] + (state_token_alias or STATE_TOKEN)
    state_pos = len(tokenizer(through_state, add_special_tokens=False)["input_ids"]) - 1
    if history_start >= state_pos:
        raise ValueError(f"Invalid bottleneck positions: history_start={history_start}, state_pos={state_pos}")
    return history_start, state_pos


def encode_example(
    tokenizer,
    row: dict[str, Any],
    max_length: int,
    supervise_eos: bool,
    bottleneck_mask: bool = False,
    state_token_alias: str | None = None,
) -> dict[str, list[int] | int]:
    raw_prefix = str(row["prompt"])
    prefix = encode_prompt_text(raw_prefix, state_token_alias)
    target = str(row["target"])
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    history_start, state_pos = (-1, -1)
    if bottleneck_mask:
        history_start, state_pos = find_bottleneck_positions(tokenizer, raw_prefix, state_token_alias)
    if not target_ids:
        raise ValueError(f"empty target for {row.get('example_id')}")
    add_eos = tokenizer.eos_token_id is not None
    max_prefix = max_length - len(target_ids) - int(add_eos)
    if max_prefix <= 0:
        raise ValueError(f"target too long for max_length={max_length}: {target}")
    if len(prefix_ids) > max_prefix:
        trim = len(prefix_ids) - max_prefix
        prefix_ids = prefix_ids[-max_prefix:]
        if bottleneck_mask:
            history_start -= trim
            state_pos -= trim
            if history_start < 0 or state_pos < 0:
                raise ValueError(f"Trimming removed bottleneck positions for {row.get('example_id')}")
    input_ids = prefix_ids + target_ids
    labels = [-100] * len(prefix_ids) + target_ids
    if add_eos:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id if supervise_eos else -100)
    out: dict[str, list[int] | int] = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    if bottleneck_mask:
        out["history_start"] = history_start
        out["state_pos"] = state_pos
        out["prompt_end"] = len(prefix_ids)
    return out


def build_bottleneck_attention_mask(
    attention_2d: torch.Tensor,
    history_starts: torch.Tensor,
    state_positions: torch.Tensor,
    prompt_ends: torch.Tensor | None = None,
    style: str = "standard",
) -> torch.Tensor:
    if style not in BOTTLENECK_STYLES:
        raise ValueError(f"Unknown bottleneck style: {style}")
    batch_size, seq_len = attention_2d.shape
    device = attention_2d.device
    min_value = torch.finfo(torch.float32).min
    causal_block = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    mask.masked_fill_(causal_block.view(1, 1, seq_len, seq_len), min_value)
    pad_cols = attention_2d.eq(0).view(batch_size, 1, 1, seq_len)
    mask.masked_fill_(pad_cols, min_value)
    for b in range(batch_size):
        h0 = int(history_starts[b].item())
        state_pos = int(state_positions[b].item())
        valid_len = int(attention_2d[b].sum().item())
        if not (0 <= h0 < state_pos < valid_len):
            raise ValueError(f"Invalid mask positions: h0={h0}, state_pos={state_pos}, valid_len={valid_len}")
        # Tokens after STATE must not attend to the history segment. The STATE
        # row itself remains unmasked so it can aggregate history.
        mask[b, 0, state_pos + 1 : valid_len, h0:state_pos] = min_value
        if style == "late_fusion":
            if prompt_ends is None:
                prompt_end = valid_len
            else:
                prompt_end = int(prompt_ends[b].item())
            if not (state_pos < prompt_end <= valid_len):
                raise ValueError(
                    f"Invalid prompt_end for late_fusion: state_pos={state_pos}, "
                    f"prompt_end={prompt_end}, valid_len={valid_len}"
                )
            # Keep question tokens pure, but open the final prompt token (the
            # answer anchor). Its logits predict target[0] in a decoder-only LM.
            answer_anchor = prompt_end - 1
            if answer_anchor < state_pos:
                raise ValueError(
                    f"Invalid answer anchor: state_pos={state_pos}, "
                    f"prompt_end={prompt_end}"
                )
            mask[b, 0, state_pos + 1 : answer_anchor, state_pos] = min_value
    return mask


def build_causal_4d_attention_mask(attention_2d: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_2d.shape
    device = attention_2d.device
    min_value = torch.finfo(torch.float32).min
    causal_block = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    mask.masked_fill_(causal_block.view(1, 1, seq_len, seq_len), min_value)
    pad_cols = attention_2d.eq(0).view(batch_size, 1, 1, seq_len)
    mask.masked_fill_(pad_cols, min_value)
    return mask


def assert_bottleneck_mask(
    mask: torch.Tensor,
    history_start: int,
    state_pos: int,
    valid_len: int,
    prompt_end: int | None = None,
    style: str = "standard",
) -> None:
    blocked = mask[0, 0, state_pos + 1, history_start:state_pos]
    if not torch.isneginf(blocked).all() and not (blocked < -1e20).all():
        raise AssertionError("post-STATE row can still attend to history")
    allowed = mask[0, 0, state_pos, history_start:state_pos]
    if not torch.all(allowed == 0):
        raise AssertionError("STATE row cannot attend to history")
    future = mask[0, 0, :valid_len, :valid_len].triu(diagonal=1)
    if not torch.all((future == 0) | (future < -1e20)):
        raise AssertionError("mask has unexpected positive values")
    if style == "late_fusion":
        if prompt_end is None:
            raise AssertionError("late_fusion assertion requires prompt_end")
        # ``prompt_end - 1`` is the answer anchor whose logits predict the
        # first supervised token.  Late fusion intentionally opens that row to
        # STATE; only the preceding question rows must remain state-blind.
        answer_anchor = prompt_end - 1
        q_blocked = mask[0, 0, state_pos + 1 : answer_anchor, state_pos]
        if q_blocked.numel() and not (q_blocked < -1e20).all():
            raise AssertionError("question rows can still attend to STATE")
        if mask[0, 0, answer_anchor, state_pos] != 0:
            raise AssertionError("answer anchor cannot attend to STATE")
        if prompt_end < valid_len and mask[0, 0, prompt_end, state_pos] != 0:
            raise AssertionError("target rows cannot attend to STATE")


class BottleneckCollator:
    def __init__(self, tokenizer, style: str = "standard", mask_prob: float = 1.0) -> None:
        self.tokenizer = tokenizer
        self.style = style
        self.mask_prob = mask_prob
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if not (0.0 <= self.mask_prob <= 1.0):
            raise ValueError(f"mask_prob must be in [0, 1], got {self.mask_prob}")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention, labels, history_starts, state_positions, prompt_ends = [], [], [], [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            labs = list(feature["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)
            labels.append(labs + [-100] * pad)
            history_starts.append(int(feature["history_start"]))
            state_positions.append(int(feature["state_pos"]))
            prompt_ends.append(int(feature["prompt_end"]))
        attention_2d = torch.tensor(attention, dtype=torch.long)
        history_tensor = torch.tensor(history_starts, dtype=torch.long)
        state_tensor = torch.tensor(state_positions, dtype=torch.long)
        prompt_end_tensor = torch.tensor(prompt_ends, dtype=torch.long)
        attention_mask = build_bottleneck_attention_mask(
            attention_2d, history_tensor, state_tensor, prompt_end_tensor, self.style
        )
        if self.mask_prob < 1.0:
            causal_mask = build_causal_4d_attention_mask(attention_2d)
            if self.mask_prob <= 0.0:
                attention_mask = causal_mask
            else:
                use_bottleneck = torch.rand(len(features), dtype=torch.float32).lt(self.mask_prob)
                selector = use_bottleneck.view(-1, 1, 1, 1).to(attention_mask.device)
                attention_mask = torch.where(selector, attention_mask, causal_mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class C1Dataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int,
        limit: int | None,
        supervise_eos: bool,
        task_filter: str | None = None,
        exclude_task: str | None = None,
        bottleneck_mask: bool = False,
        state_token_alias: str | None = None,
    ) -> None:
        self.rows = read_jsonl(path, limit, task_filter, exclude_task)
        self.examples = [
            encode_example(tokenizer, row, max_length, supervise_eos, bottleneck_mask, state_token_alias)
            for row in self.rows
        ]
        self.examples = [ex for ex in self.examples if any(label != -100 for label in ex["labels"])]
        if not self.examples:
            raise ValueError(f"No trainable examples from {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


class StateTokenTrainer(Trainer):
    """Trainer that keeps PEFT trainable-token checkpoints row-sparse."""

    def __init__(self, *args, save_embedding_layers: bool | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_embedding_layers = save_embedding_layers

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        if self.save_embedding_layers is not False:
            super()._save(output_dir, state_dict)
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            save_embedding_layers=False,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path)
    parser.add_argument("--task-filter")
    parser.add_argument("--exclude-task")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-valid-samples", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        help="Use an exact epoch count; requires --max-steps -1.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--add-state-token", action="store_true")
    parser.add_argument("--state-token", default=STATE_TOKEN)
    parser.add_argument(
        "--train-state-token-only",
        action="store_true",
        help=(
            "With --add-state-token, train and save only the new state-token embedding row via "
            "PEFT trainable_token_indices instead of saving the full embedding matrix."
        ),
    )
    parser.add_argument(
        "--state-token-alias",
        help=f"Reuse an existing single tokenizer token for {STATE_TOKEN}, e.g. {DEFAULT_STATE_TOKEN_ALIAS}",
    )
    parser.add_argument("--bottleneck-mask", action="store_true")
    parser.add_argument("--bottleneck-style", choices=BOTTLENECK_STYLES, default="standard")
    parser.add_argument(
        "--bottleneck-mask-prob",
        type=float,
        default=1.0,
        help="When bottleneck mask is enabled, randomly apply it to this fraction of training examples; "
        "the rest receive a standard causal 4D mask.",
    )
    parser.add_argument("--save-final-only", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of Trainer checkpoints to retain; use a larger value for checkpoint curves.",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help="Initialize a new optimizer run from an existing LoRA adapter.",
    )
    parser.add_argument("--supervise-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.add_state_token and args.state_token_alias:
        raise ValueError("Use either --add-state-token or --state-token-alias, not both")
    if args.task_filter and args.exclude_task:
        raise ValueError("Use either --task-filter or --exclude-task, not both")
    if args.train_state_token_only and not args.add_state_token:
        raise ValueError("--train-state-token-only requires --add-state-token")
    # A PEFT Trainer checkpoint stores optimizer/scheduler state and adapter
    # weights, but reconstructing a fresh adapter from the CLI defaults can
    # silently change its rank.  When both are provided, initialize the exact
    # adapter architecture from ``--init-adapter`` and then let Trainer restore
    # the checkpoint weights and training state.
    if args.num_train_epochs is not None:
        if args.num_train_epochs <= 0 or args.max_steps != -1:
            raise ValueError("--num-train-epochs requires a positive value and --max-steps -1")

    # Trainer seeds its dataloader in ``Trainer.__init__``, but LoRA modules are
    # created below before the Trainer exists. Seed here as well so ``--seed``
    # controls both LoRA initialization and subsequent sampling.
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_tokenizer_len = len(tokenizer)
    added_state_token = False
    state_id: int | None = None
    if args.add_state_token:
        added_state_token = add_special_state_token(tokenizer, args.state_token)
        state_id = tokenizer(args.state_token, add_special_tokens=False)["input_ids"][0]
        print(
            f"[train] state_token={args.state_token} state_id={state_id} "
            f"tokenizer_len={len(tokenizer)} added={added_state_token}",
            flush=True,
        )
    elif args.state_token_alias:
        alias_ids = tokenizer(args.state_token_alias, add_special_tokens=False)["input_ids"]
        if len(alias_ids) != 1:
            raise ValueError(f"--state-token-alias must tokenize to one id, got {alias_ids}")
        print(
            f"[train] state_token={STATE_TOKEN} alias={args.state_token_alias} "
            f"state_id={alias_ids[0]} tokenizer_len={len(tokenizer)} added=False",
            flush=True,
        )

    train_ds = C1Dataset(
        args.train_jsonl,
        tokenizer,
        args.max_length,
        args.max_train_samples,
        args.supervise_eos,
        args.task_filter,
        args.exclude_task,
        args.bottleneck_mask,
        args.state_token_alias,
    )
    valid_ds = None
    if args.valid_jsonl:
        valid_ds = C1Dataset(
            args.valid_jsonl,
            tokenizer,
            args.max_length,
            args.max_valid_samples,
            args.supervise_eos,
            args.task_filter,
            args.exclude_task,
            args.bottleneck_mask,
            args.state_token_alias,
        )

    quant = None
    if args.load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs = {}
    if args.bottleneck_mask:
        model_kwargs["attn_implementation"] = "eager"
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            os.environ.get(
                "OMPI_COMM_WORLD_LOCAL_RANK",
                os.environ.get("MPI_LOCALRANKID", os.environ.get("MV2_COMM_WORLD_LOCAL_RANK", "-1")),
            ),
        )
    )
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device_map: str | dict[str, int] | None = {"": local_rank}
    else:
        device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        **model_kwargs,
    )
    resize_and_mean_init_new_tokens(model, len(tokenizer), base_tokenizer_len)
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    modules_to_save = ["embed_tokens"] if args.add_state_token and not args.train_state_token_only else None
    trainable_token_indices = [state_id] if args.train_state_token_only and state_id is not None else None
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        modules_to_save=modules_to_save,
        trainable_token_indices=trainable_token_indices,
    )
    if args.init_adapter:
        from peft import PeftModel

        weights = args.init_adapter / "adapter_model.safetensors"
        config = args.init_adapter / "adapter_config.json"
        if not weights.is_file() or not config.is_file():
            raise FileNotFoundError(f"Incomplete init adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)
    # The frozen backbone embeddings otherwise produce inputs without a grad
    # function.  Reentrant gradient checkpointing then drops the LoRA path and
    # fails at the first backward pass when warm-starting from a PEFT adapter.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    if args.bottleneck_mask:
        first = train_ds[0]
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
            int(sum(first["attention_mask"])),
            int(first["prompt_end"]),
            args.bottleneck_style,
        )
        print(
            f"[train] bottleneck mask enabled: style={args.bottleneck_style} "
            f"mask_prob={args.bottleneck_mask_prob} "
            f"history_start={first['history_start']} state_pos={first['state_pos']} "
            f"prompt_end={first['prompt_end']}",
            flush=True,
        )
        collator = BottleneckCollator(tokenizer, args.bottleneck_style, args.bottleneck_mask_prob)
    else:
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        seed=args.seed,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs or 1.0,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=max(1, args.max_steps // 20),
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if valid_ds is not None else "no",
        eval_steps=(args.eval_steps or max(25, args.max_steps // 4)) if valid_ds is not None else None,
        save_strategy="no" if args.save_final_only else "steps",
        save_steps=args.save_steps or max(50, args.max_steps // 2),
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))) > 1 else None,
    )
    trainer = StateTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        processing_class=tokenizer,
        save_embedding_layers=False if args.train_state_token_only else None,
    )
    if args.resume_from_checkpoint:
        with non_tensor_parallel_peft_resume(model):
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=None)
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    meta = {
        "model": args.model,
        "train_jsonl": str(args.train_jsonl),
        "valid_jsonl": str(args.valid_jsonl) if args.valid_jsonl else None,
        "task_filter": args.task_filter,
        "exclude_task": args.exclude_task,
        "max_train_samples": args.max_train_samples,
        "max_valid_samples": args.max_valid_samples,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "seed": args.seed,
        "add_state_token": args.add_state_token,
        "train_state_token_only": args.train_state_token_only,
        "state_token": args.state_token if args.add_state_token else None,
        "state_token_id": state_id,
        "state_token_alias": args.state_token_alias,
        "added_state_token": added_state_token,
        "tokenizer_len": len(tokenizer),
        "save_final_only": args.save_final_only,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "bottleneck_mask": args.bottleneck_mask,
        "bottleneck_style": args.bottleneck_style if args.bottleneck_mask else None,
        "bottleneck_mask_prob": args.bottleneck_mask_prob if args.bottleneck_mask else None,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
