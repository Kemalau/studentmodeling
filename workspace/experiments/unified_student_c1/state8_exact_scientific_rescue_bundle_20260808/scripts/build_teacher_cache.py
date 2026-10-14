#!/usr/bin/env python3
"""Build an immutable full-context teacher-logit cache for M3/M4.

The cache stores distributions only at supervised target positions. This
avoids a second adapter forward during every recurrent optimizer step while
preserving target-token alignment across different prompt lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Callable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from recurrent_student_model import expand_state_token_alias
from state8_natural_curriculum_view import apply_state8_natural_curriculum_view
from train_recurrent_student_diagnostic import alias_text, encode_prompt_target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_partition_indices(
    row_count: int,
    *,
    order_seed: int | None,
    partition_index: int,
    partition_count: int,
) -> list[int]:
    if row_count <= 0:
        raise ValueError("Teacher cache source has no rows")
    if partition_count <= 0 or not 0 <= partition_index < partition_count:
        raise ValueError("Invalid teacher-cache partition")
    order = list(range(row_count))
    if order_seed is not None:
        random.Random(order_seed).shuffle(order)
    return order[partition_index::partition_count]


def jsonl_offsets(path: Path) -> list[int]:
    offsets = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def resumable_shard_prefix(
    output_dir: Path,
    *,
    selected_indices: list[int],
    shard_size: int,
    top_k: int,
    expected_record: Callable[[int, int], tuple[str, list[int]]],
) -> tuple[list[dict], int]:
    """Validate and reconstruct the largest atomically saved cache prefix.

    A cache shard is renamed from ``.partial`` only after ``torch.save`` has
    completed.  This lets a preempted Feeds attempt reuse every complete shard
    while rebuilding the in-memory index deterministically.  Validation is
    deliberately strict: filenames must be contiguous and every stored
    example, target-token sequence, and top-k tensor must match the current
    source/configuration before any work is skipped.
    """

    pattern = re.compile(r"shard-(\d{5})\.pt")
    numbered: list[tuple[int, Path]] = []
    for path in output_dir.glob("shard-*.pt"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected teacher-cache shard name: {path.name}")
        numbered.append((int(match.group(1)), path))
    numbered.sort()
    if [number for number, _ in numbered] != list(range(len(numbered))):
        raise ValueError("Teacher-cache resume shards are not a contiguous prefix")

    index: list[dict] = []
    for shard_id, path in numbered:
        start = shard_id * shard_size
        if start >= len(selected_indices):
            raise ValueError(f"Teacher-cache shard exceeds selected rows: {path.name}")
        expected_count = min(shard_size, len(selected_indices) - start)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or len(payload) != expected_count:
            raise ValueError(
                f"Teacher-cache resume shard size mismatch: {path.name} "
                f"rows={len(payload) if isinstance(payload, dict) else 'invalid'} "
                f"expected={expected_count}"
            )
        for offset, (key, record) in enumerate(payload.items()):
            position = start + offset
            row_index = selected_indices[position]
            example_id, target_tokens = expected_record(position, row_index)
            if str(key) != example_id or str(record.get("example_id")) != example_id:
                raise ValueError(
                    f"Teacher-cache resume example mismatch at partition position {position}"
                )
            stored_tokens = [int(value) for value in record.get("target_token_ids", [])]
            if stored_tokens != target_tokens:
                raise ValueError(
                    f"Teacher-cache resume target-token mismatch for {example_id}"
                )
            ids = record.get("teacher_topk_ids")
            logprobs = record.get("teacher_topk_logprobs")
            expected_shape = (len(target_tokens), top_k)
            if (
                not isinstance(ids, torch.Tensor)
                or not isinstance(logprobs, torch.Tensor)
                or tuple(ids.shape) != expected_shape
                or tuple(logprobs.shape) != expected_shape
                or not torch.isfinite(logprobs.float()).all()
            ):
                raise ValueError(f"Teacher-cache resume tensor mismatch for {example_id}")
            index.append(
                {
                    "example_id": example_id,
                    "file": path.name,
                    "target_tokens": len(target_tokens),
                    "source_row_index": row_index,
                    "partition_position": position,
                }
            )
    return index, len(numbered)


def atomic_write_json(path: Path, value: object, *, sort_keys: bool = False) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prompt_for_cache(row: dict, prompt_field: str) -> str:
    """Select the explicitly declared teacher prompt without silent fallback."""

    if prompt_field == "full_context_text_or_direct":
        value = row.get("full_context_text") or row.get("direct_text")
    else:
        value = row.get(prompt_field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Teacher-cache row {row.get('example_id')} lacks {prompt_field}"
        )
    return value


def prepare_cache_row(row: dict, *, state8_natural_view: bool) -> dict:
    """Apply the same train-only target/prompt view used by State8 training."""

    value = dict(row)
    if state8_natural_view:
        apply_state8_natural_curriculum_view([value])
    return value


def cache_build_spec(
    *,
    model: str,
    teacher_adapter: Path,
    teacher_adapter_sha256: str,
    source_jsonl: Path,
    prompt_field: str,
    state_token_alias: str,
    state_token_count: int,
    state8_natural_view: bool,
    max_length: int,
    top_k: int,
    shard_size: int,
    batch_size: int,
    attn_implementation: str,
    logprob_dtype: str,
    layerwise_online_kd: bool,
    order_seed: int | None,
    partition_index: int,
    partition_count: int,
) -> dict:
    view_path = Path(__file__).resolve().parent / "state8_natural_curriculum_view.py"
    return {
        "version": 1,
        "cache_role": "online_kd" if layerwise_online_kd else "behavior_kd",
        "model": model,
        "teacher_adapter": str(teacher_adapter),
        "teacher_adapter_sha256": teacher_adapter_sha256,
        "source_jsonl": str(source_jsonl),
        "source_sha256": sha256(source_jsonl),
        "prompt_field": prompt_field,
        "state_token_alias": state_token_alias,
        "state_token_count": state_token_count,
        "state_token_rendered_alias": expand_state_token_alias(
            state_token_alias, state_token_count
        ),
        "state8_natural_view": state8_natural_view,
        "state8_natural_view_sha256": (
            sha256(view_path) if state8_natural_view else None
        ),
        "max_length": max_length,
        "top_k": top_k,
        "shard_size": shard_size,
        "batch_size": batch_size,
        "attn_implementation": attn_implementation,
        "dtype": logprob_dtype,
        "layerwise_online_kd": layerwise_online_kd,
        "order_seed": order_seed,
        "partition_index": partition_index,
        "partition_count": partition_count,
    }


def bind_build_spec(output_dir: Path, spec: dict) -> None:
    """Pin resumable shards to one complete scientific cache recipe."""

    path = output_dir / "build_spec.json"
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != spec:
            raise ValueError("Teacher-cache resumable build-spec drift")
        return
    atomic_write_json(path, spec, sort_keys=True)


@torch.no_grad()
def batched_target_logprobs(
    model,
    encoded_rows: list[dict[str, torch.Tensor]],
    top_k: int,
    *,
    pad_token_id: int,
    logprob_dtype: torch.dtype = torch.float16,
    renormalize_topk: bool = False,
) -> list[tuple[list[int], torch.Tensor, torch.Tensor]]:
    """Return per-example teacher distributions from one padded forward pass.

    Padding is applied only on the right and labels remain ``-100`` there, so
    each returned distribution is aligned to exactly the same supervised
    target positions as the historical single-example implementation.
    """

    if not encoded_rows:
        raise ValueError("Teacher-cache batch is empty")
    max_length = max(int(row["input_ids"].shape[1]) for row in encoded_rows)
    batch_size = len(encoded_rows)
    input_ids = torch.full(
        (batch_size, max_length), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    for index, row in enumerate(encoded_rows):
        length = int(row["input_ids"].shape[1])
        input_ids[index, :length] = row["input_ids"][0]
        attention_mask[index, :length] = row["attention_mask"][0]
        labels[index, :length] = row["labels"][0]

    device = next(model.parameters()).device
    inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }
    if isinstance(model, LayerwiseRecurrentStudentModel):
        output = model.capture_direct_state(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        ).model_output
    else:
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
    results = []
    for index in range(batch_size):
        valid = inputs["labels"][index, 1:].ne(-100)
        tokens = inputs["labels"][index, 1:][valid]
        answer_logits = output.logits[index, :-1].float()[valid]
        k = min(int(top_k), answer_logits.shape[-1])
        values, indices = torch.topk(answer_logits, k=k, dim=-1, sorted=True)
        if renormalize_topk:
            # Match ``detached_teacher_topk`` exactly: take top-k raw logits,
            # then normalize inside that support.  Behavior-KD legacy caches
            # retain their historical full-vocabulary normalization below.
            values = torch.log_softmax(values, dim=-1)
        else:
            full_logprobs = torch.log_softmax(answer_logits, dim=-1)
            values = full_logprobs.gather(-1, indices)
        results.append(
            (
                tokens.cpu().tolist(),
                indices.to(dtype=torch.int32, device="cpu"),
                values.to(dtype=logprob_dtype, device="cpu"),
            )
        )
    return results


@torch.no_grad()
def target_logprobs(
    model, encoded: dict[str, torch.Tensor], top_k: int
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Backward-compatible single-example wrapper used by existing callers."""

    return batched_target_logprobs(
        model,
        [encoded],
        top_k,
        pad_token_id=0,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher-adapter", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--state-token-count", type=int, default=1)
    parser.add_argument(
        "--prompt-field",
        choices=("full_context_text_or_direct", "full_context_text", "direct_text"),
        default="full_context_text_or_direct",
    )
    parser.add_argument("--state8-natural-view", action="store_true")
    parser.add_argument(
        "--layerwise-online-kd",
        action="store_true",
        help=(
            "Use the same layerwise State8 direct forward and SDPA mask as "
            "qualified online answer-logit distillation."
        ),
    )
    parser.add_argument("--teacher-adapter-sha256")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument(
        "--logprob-dtype", choices=("float16", "float32"), default="float16"
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.state_token_count <= 0:
        raise ValueError("--state-token-count must be positive")
    if args.layerwise_online_kd and (
        args.prompt_field != "direct_text"
        or not args.state8_natural_view
        or args.state_token_count != 8
        or args.attn_implementation != "sdpa"
        or args.logprob_dtype != "float32"
    ):
        raise ValueError(
            "Layerwise online-KD cache requires direct_text, the State8 natural "
            "view, eight carriers, SDPA, and float32 log-probabilities"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Teacher cache requires CUDA")
    offsets = jsonl_offsets(args.jsonl)
    teacher_weights = args.teacher_adapter / "adapter_model.safetensors"
    if not teacher_weights.is_file():
        raise FileNotFoundError(f"Teacher adapter is incomplete: {args.teacher_adapter}")
    teacher_digest = sha256(teacher_weights)
    if (
        args.teacher_adapter_sha256 is not None
        and teacher_digest != str(args.teacher_adapter_sha256).lower()
    ):
        raise ValueError("Teacher-cache adapter SHA256 mismatch")
    rendered_alias = expand_state_token_alias(
        args.state_token_alias, args.state_token_count
    )
    spec = cache_build_spec(
        model=str(args.model),
        teacher_adapter=args.teacher_adapter,
        teacher_adapter_sha256=teacher_digest,
        source_jsonl=args.jsonl,
        prompt_field=args.prompt_field,
        state_token_alias=args.state_token_alias,
        state_token_count=args.state_token_count,
        state8_natural_view=args.state8_natural_view,
        max_length=args.max_length,
        top_k=args.top_k,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        attn_implementation=args.attn_implementation,
        logprob_dtype=args.logprob_dtype,
        layerwise_online_kd=args.layerwise_online_kd,
        order_seed=args.order_seed,
        partition_index=args.partition_index,
        partition_count=args.partition_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bind_build_spec(args.output_dir, spec)
    selected_indices = ordered_partition_indices(
        len(offsets),
        order_seed=args.order_seed,
        partition_index=args.partition_index,
        partition_count=args.partition_count,
    )
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for partial in args.output_dir.glob("*.partial"):
        partial.unlink()

    with args.jsonl.open("rb") as resume_source:
        def expected_record(position: int, row_index: int) -> tuple[str, list[int]]:
            del position
            resume_source.seek(offsets[row_index])
            row = prepare_cache_row(
                json.loads(resume_source.readline()),
                state8_natural_view=args.state8_natural_view,
            )
            prompt = prompt_for_cache(row, args.prompt_field)
            encoded = encode_prompt_target(
                tokenizer,
                alias_text(prompt, rendered_alias),
                str(row["target"]),
                args.max_length,
            )
            labels = encoded["labels"][0, 1:]
            target_tokens = labels[labels.ne(-100)].cpu().tolist()
            return str(row["example_id"]), [int(value) for value in target_tokens]

        index, shard_id = resumable_shard_prefix(
            args.output_dir,
            selected_indices=selected_indices,
            shard_size=args.shard_size,
            top_k=args.top_k,
            expected_record=expected_record,
        )
    completed_rows = len(index)
    if completed_rows:
        print(
            json.dumps(
                {
                    "resumed_cached_rows": completed_rows,
                    "remaining_rows": len(selected_indices) - completed_rows,
                }
            ),
            flush=True,
        )

    model = None
    if completed_rows < len(selected_indices):
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
            attn_implementation=args.attn_implementation,
        )
        model = PeftModel.from_pretrained(
            base, args.teacher_adapter, is_trainable=False
        ).eval()
        if args.layerwise_online_kd:
            alias_ids = tokenizer(
                args.state_token_alias, add_special_tokens=False
            )["input_ids"]
            if len(alias_ids) != 1:
                raise ValueError("State8 online-KD alias must tokenize to one ID")
            model = LayerwiseRecurrentStudentModel(
                model,
                state_token_id=int(alias_ids[0]),
                transition_rank=0,
                operation_checkpointing=False,
                state_token_count=args.state_token_count,
            ).eval()

    shard_payload: dict[str, dict] = {}

    def flush_shard() -> None:
        nonlocal shard_id, shard_payload
        if not shard_payload:
            return
        filename = f"shard-{shard_id:05d}.pt"
        temporary = args.output_dir / f"{filename}.partial"
        torch.save(shard_payload, temporary)
        temporary.replace(args.output_dir / filename)
        shard_id += 1
        shard_payload = {}

    with args.jsonl.open("rb") as source:
        for batch_start in range(completed_rows, len(selected_indices), args.batch_size):
            batch_metadata = []
            batch_encoded = []
            batch_stop = min(batch_start + args.batch_size, len(selected_indices))
            for position in range(batch_start, batch_stop):
                row_index = selected_indices[position]
                source.seek(offsets[row_index])
                row = prepare_cache_row(
                    json.loads(source.readline()),
                    state8_natural_view=args.state8_natural_view,
                )
                prompt = prompt_for_cache(row, args.prompt_field)
                batch_encoded.append(
                    encode_prompt_target(
                        tokenizer,
                        alias_text(prompt, rendered_alias),
                        str(row["target"]),
                        args.max_length,
                    )
                )
                batch_metadata.append((position, row_index, str(row["example_id"])))
            assert model is not None
            batch_outputs = batched_target_logprobs(
                model,
                batch_encoded,
                args.top_k,
                pad_token_id=int(tokenizer.pad_token_id),
                logprob_dtype=(
                    torch.float32
                    if args.logprob_dtype == "float32"
                    else torch.float16
                ),
                renormalize_topk=args.layerwise_online_kd,
            )
            for (position, row_index, example_id), output in zip(
                batch_metadata, batch_outputs, strict=True
            ):
                tokens, topk_ids, topk_logprobs = output
                filename = f"shard-{shard_id:05d}.pt"
                shard_payload[example_id] = {
                    "example_id": example_id,
                    "target_token_ids": tokens,
                    "teacher_topk_ids": topk_ids,
                    "teacher_topk_logprobs": topk_logprobs,
                }
                index.append(
                    {
                        "example_id": example_id,
                        "file": filename,
                        "target_tokens": len(tokens),
                        "source_row_index": row_index,
                        "partition_position": position,
                    }
                )
                if len(shard_payload) >= args.shard_size:
                    flush_shard()
                if position == 0 or (position + 1) % 100 == 0:
                    print(json.dumps({"cached": position + 1}), flush=True)

    flush_shard()

    index_path = args.output_dir / "index.json"
    atomic_write_json(index_path, index)
    manifest = {
        "version": 1,
        "status": "complete",
        "model": args.model,
        "teacher_adapter": str(args.teacher_adapter),
        "teacher_adapter_sha256": teacher_digest,
        "source_jsonl": str(args.jsonl),
        "source_sha256": sha256(args.jsonl),
        "rows": len(selected_indices),
        "source_rows": len(offsets),
        "order_seed": args.order_seed,
        "partition_index": args.partition_index,
        "partition_count": args.partition_count,
        "state_token_alias": args.state_token_alias,
        "state_token_count": args.state_token_count,
        "state_token_rendered_alias": rendered_alias,
        "prompt_field": args.prompt_field,
        "state8_natural_view": args.state8_natural_view,
        "state8_natural_view_sha256": spec["state8_natural_view_sha256"],
        "cache_role": spec["cache_role"],
        "layerwise_online_kd": args.layerwise_online_kd,
        "max_length": args.max_length,
        "dtype": args.logprob_dtype,
        "top_k": args.top_k,
        "shard_size": args.shard_size,
        "batch_size": args.batch_size,
        "attn_implementation": args.attn_implementation,
        "shards": shard_id,
        "scope": (
            "renormalized_topk_logits_at_supervised_target_positions"
            if args.layerwise_online_kd
            else "topk_logprobs_at_supervised_target_positions"
        ),
        "index_sha256": sha256(index_path),
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
