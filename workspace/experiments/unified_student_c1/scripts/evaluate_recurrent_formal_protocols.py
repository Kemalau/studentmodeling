#!/usr/bin/env python3
"""Evaluate formal protocols through the closed recurrent student state only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_c1 import load_model
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from evaluate_formal_protocols import (
    BINARY_TASKS,
    GENERATION_TASKS,
    redacted_records,
    sample_rows,
    summarize,
    write_jsonl,
)
from recurrent_student_model import (
    RecurrentStudentModel,
    build_update_text,
    expand_state_token_alias,
    extract_direct_decode_text,
)
from train_recurrent_student_diagnostic import (
    alias_text,
    encode_prompt,
    encode_prompt_target,
    move,
    sequence_logprob,
)


def prepare_recurrent_modules_for_inference(wrapper, backbone) -> None:
    """Move lightweight recurrent modules without lowering FP32 projections."""
    embedding_weight = backbone.get_input_embeddings().weight
    device = embedding_weight.device
    for name in (
        "state_norm",
        "state_in_proj",
        "state_out_proj",
        "transition_in",
        "transition_out",
    ):
        module = getattr(wrapper, name, None)
        if module is not None:
            module.to(device=device)


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_wrapper(
    model_path: str,
    adapter_path: str,
    recurrent_state_path: Path | None,
    alias: str,
    state_architecture: str = "scalar",
):
    tokenizer, backbone = load_model(model_path, adapter_path, bottleneck_mask=True, load_4bit=True)
    backbone.config._attn_implementation = "sdpa"
    ids = tokenizer(alias, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"State alias must be one token, got {ids}")
    if state_architecture == "scalar":
        if recurrent_state_path is None:
            raise ValueError("Scalar state evaluation requires --recurrent-state")
        payload = torch.load(
            recurrent_state_path, map_location="cpu", weights_only=False
        )
        config = payload.get("config") or {}
        trainable = bool(config.get("trainable_projections"))
        wrapper = RecurrentStudentModel(
            backbone,
            state_token_id=int(ids[0]),
            trainable_projections=trainable,
            state_projection_rank=int(config.get("state_projection_rank", 64)),
        )
        wrapper.state_in_proj.load_state_dict(payload["state_in_proj"])
        wrapper.state_out_proj.load_state_dict(payload["state_out_proj"])
    elif state_architecture == "layerwise":
        transition_rank = 0
        state_token_count = 1
        if recurrent_state_path is not None:
            payload = torch.load(
                recurrent_state_path, map_location="cpu", weights_only=False
            )
            config = payload.get("config") or {}
            recorded = config.get("state_architecture", "layerwise")
            if recorded != "layerwise":
                raise ValueError(
                    "Layerwise state artifact has incompatible architecture: "
                    f"{recorded}"
                )
            transition_rank = int(config.get("layerwise_transition_rank", 0))
            state_token_count = int(config.get("state_token_count", 1))
        wrapper = LayerwiseRecurrentStudentModel(
            backbone,
            int(ids[0]),
            transition_rank=transition_rank,
            state_token_count=state_token_count,
        )
        if recurrent_state_path is not None:
            if "transition_in" not in payload or "transition_out" not in payload:
                raise ValueError("Layerwise state artifact lacks transition weights")
            wrapper.transition_in.load_state_dict(payload["transition_in"])
            wrapper.transition_out.load_state_dict(payload["transition_out"])
    else:
        raise ValueError(f"Unknown state architecture: {state_architecture}")
    # Projection weights remain FP32; ResidualLowRankProjection handles the
    # BF16 state boundary explicitly in both training and inference.
    prepare_recurrent_modules_for_inference(wrapper, backbone)
    wrapper.state_token_rendered_alias = expand_state_token_alias(
        alias, int(getattr(wrapper, "state_token_count", 1))
    )
    wrapper.eval()
    return tokenizer, wrapper


def activate_decode_adapter(wrapper, capture_adapter: str, decode_adapter: str) -> str:
    """Switch adapters only after recurrent states have been constructed."""
    if decode_adapter == capture_adapter:
        return "default"
    backbone = wrapper.backbone
    if not hasattr(backbone, "load_adapter") or not hasattr(backbone, "set_adapter"):
        raise TypeError("Cross-adapter recurrent evaluation requires a PEFT model")
    name = "closed_state_decoder"
    backbone.load_adapter(decode_adapter, adapter_name=name, is_trainable=False)
    backbone.set_adapter(name)
    backbone.eval()
    return name


def evaluation_metadata(row: dict) -> dict:
    """Recover frozen formal metadata from raw or recurrent-manifest rows."""
    metadata = row.get("metadata") or row.get("original_metadata") or {}
    return {
        "source": str(metadata.get("eval_source") or row.get("source") or "unknown"),
        "protocol": str(metadata.get("eval_protocol") or "unknown"),
        "delta": metadata.get("prospective_delta"),
    }


def update_texts_for_row(row: dict) -> list[str]:
    """Use audited update texts when present; fall back only for raw rows."""
    if "update_texts" in row:
        return [str(value) for value in row.get("update_texts") or []]
    return [build_update_text(interaction) for interaction in row.get("history") or []]


def evaluation_shard(
    rows: list[dict[str, Any]], shard_index: int | None, num_shards: int | None
) -> list[dict[str, Any]]:
    """Return one stable, disjoint evaluation shard.

    Sharding happens after task/source filtering but before any optional group
    sampling.  Sorting by the immutable example ID makes the partition
    independent of JSONL ordering; round-robin assignment keeps shard sizes
    within one row.  The merge audit reconstructs the same ID partition and
    rejects gaps or duplicates before corpus metrics are computed.
    """
    if shard_index is None and num_shards is None:
        return rows
    if shard_index is None or num_shards is None:
        raise ValueError("--shard-index and --num-shards must be provided together")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Evaluation shard must satisfy 0 <= index < num_shards")
    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    if len({str(row["example_id"]) for row in ordered}) != len(ordered):
        raise ValueError("Evaluation rows contain duplicate example IDs")
    return [row for position, row in enumerate(ordered) if position % num_shards == shard_index]


@torch.inference_mode()
def recurrent_state(wrapper, tokenizer, row: dict, alias: str, max_length: int) -> torch.Tensor:
    device = next(wrapper.parameters()).device
    state = wrapper.initial_state(1, device=device)
    for update_text in update_texts_for_row(row):
        update = move(
            encode_prompt(
                tokenizer,
                alias_text(update_text, alias),
                max_length,
            ),
            device,
        )
        state = wrapper.update_state(state, **update).state
    return state


def pad_prompt_batches(
    batches: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("Cannot pad an empty prompt batch")
    max_len = max(int(batch["input_ids"].shape[1]) for batch in batches)
    return {
        key: torch.cat(
            [
                torch.nn.functional.pad(
                    batch[key],
                    (0, max_len - int(batch[key].shape[1])),
                    value=(pad_token_id if key == "input_ids" else 0),
                )
                for batch in batches
            ],
            dim=0,
        )
        for key in ("input_ids", "attention_mask")
    }


@torch.inference_mode()
def recurrent_states_batched(
    wrapper,
    tokenizer,
    rows: list[dict[str, Any]],
    alias: str,
    max_length: int,
    batch_size: int,
) -> list[torch.Tensor | None]:
    """Build closed states in short padded batches, grouped by update count."""
    if batch_size <= 0:
        raise ValueError("State batch size must be positive")
    device = next(wrapper.parameters()).device
    output: list[torch.Tensor | None] = [None] * len(rows)
    by_depth: dict[int, list[int]] = {}
    updates_by_row = [update_texts_for_row(row) for row in rows]
    for index, updates in enumerate(updates_by_row):
        by_depth.setdefault(len(updates), []).append(index)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    completed = 0
    for depth, indices in sorted(by_depth.items()):
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            states = wrapper.initial_state(len(selected), device=device)
            if depth == 0 and states is None:
                # A layerwise cold start deliberately carries no fabricated
                # trajectory.  The decoder will use the pretrained carrier
                # path naturally for these rows.
                completed += len(selected)
                if completed == len(rows) or completed % 100 <= len(selected):
                    print(
                        json.dumps({"states": completed, "total": len(rows)}),
                        flush=True,
                    )
                continue
            for update_index in range(depth):
                encoded = [
                    move(
                        encode_prompt(
                            tokenizer,
                            alias_text(updates_by_row[row_index][update_index], alias),
                            max_length,
                        ),
                        device,
                    )
                    for row_index in selected
                ]
                batch = pad_prompt_batches(encoded, int(pad_id))
                states = wrapper.update_state(states, **batch).state
            if states is None:
                raise AssertionError("Non-empty update trajectory produced no state")
            for row_index, state in zip(selected, states, strict=True):
                output[row_index] = state.detach().cpu()
            completed += len(selected)
            if completed == len(rows) or completed % 100 <= len(selected):
                print(json.dumps({"states": completed, "total": len(rows)}), flush=True)
    return output


def closed_decode_text(row: dict) -> str:
    text = str(row["decode_text"]) if "decode_text" in row else extract_direct_decode_text(str(row["prompt"]))
    if "<student_history>" in text.lower():
        raise AssertionError("Closed recurrent evaluator leaked raw history")
    return text


def pad_supervised_batches(
    batches: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(int(batch["input_ids"].shape[1]) for batch in batches)
    combined = {}
    for key, value in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
        combined[key] = torch.cat(
            [
                torch.nn.functional.pad(
                    batch[key],
                    (0, max_len - int(batch[key].shape[1])),
                    value=value,
                )
                for batch in batches
            ],
            dim=0,
        )
    combined["prompt_ends"] = torch.cat(
        [batch["prompt_ends"] for batch in batches], dim=0
    )
    return combined


def sequence_logprob_at(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_end: int,
    valid_length: int,
    row_index: int,
) -> float:
    score = 0.0
    if not (0 < prompt_end <= valid_length <= input_ids.shape[1]):
        raise ValueError(
            f"Invalid candidate bounds: prompt_end={prompt_end}, "
            f"valid_length={valid_length}, padded_length={input_ids.shape[1]}"
        )
    for position in range(prompt_end, valid_length):
        log_probs = torch.log_softmax(logits[row_index, position - 1].float(), dim=-1)
        score += float(log_probs[input_ids[row_index, position]].item())
    return score


@torch.inference_mode()
def candidate_records(
    wrapper,
    tokenizer,
    rows,
    states: list[torch.Tensor | None],
    alias: str,
    max_length: int,
    row_batch_size: int,
):
    device = next(wrapper.parameters()).device
    records = []
    if row_batch_size <= 0:
        raise ValueError("Candidate row batch size must be positive")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    indexed_groups = [
        [
            (row, state)
            for row, state in zip(rows, states, strict=True)
            if (state is None) is cold
        ]
        for cold in (False, True)
    ]
    processed = 0
    for group in (group for group in indexed_groups if group):
      for start in range(0, len(group), row_batch_size):
        chunk_pairs = group[start : start + row_batch_size]
        chunk = [row for row, _state in chunk_pairs]
        chunk_states = [state for _row, state in chunk_pairs]
        encoded = []
        state_batch = []
        specs = []
        for row_offset, (row, state) in enumerate(
            zip(chunk, chunk_states, strict=True)
        ):
            text = closed_decode_text(row)
            for label_value in row["candidate_labels"]:
                label = str(label_value)
                encoded.append(
                    encode_prompt_target(
                        tokenizer,
                        alias_text(text, alias),
                        str(row["candidate_targets"][label]),
                        max_length,
                    )
                )
                state_batch.append(state)
                specs.append((row_offset, label))
        batch = move(pad_supervised_batches(encoded, int(pad_id)), device)
        prompt_ends = batch.pop("prompt_ends")
        state_argument = (
            None
            if state_batch and state_batch[0] is None
            else torch.stack(state_batch).to(device)
        )
        if any((state is None) != (state_argument is None) for state in state_batch):
            raise AssertionError("Candidate batch mixed cold and populated states")
        output = wrapper.decode_from_state(
            state_argument,
            **batch,
            prompt_ends=prompt_ends,
            output_hidden_states=False,
        )
        scores_by_row: list[dict[str, float]] = [dict() for _ in chunk]
        for candidate_index, (row_offset, label) in enumerate(specs):
            scores_by_row[row_offset][label] = sequence_logprob_at(
                output.logits,
                batch["input_ids"],
                int(prompt_ends[candidate_index].item()),
                int(batch["attention_mask"][candidate_index].sum().item()),
                candidate_index,
            )
        for row, scores in zip(chunk, scores_by_row, strict=True):
            labels = [str(label) for label in row["candidate_labels"]]
            probabilities = torch.softmax(
                torch.tensor([scores[label] for label in labels]), dim=0
            ).tolist()
            metadata = evaluation_metadata(row)
            records.append(
                {
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "source": metadata["source"],
                    "protocol": metadata["protocol"],
                    "delta": metadata["delta"],
                    "gold": str(row["target_label"]),
                    "prediction": max(scores, key=scores.get),
                    "probabilities": dict(zip(labels, probabilities)),
                }
            )
        processed += len(chunk)
        print(json.dumps({"classified": processed, "total": len(rows)}), flush=True)
    return records


@torch.inference_mode()
def generate_one(
    wrapper,
    tokenizer,
    state: torch.Tensor | None,
    decode_text: str,
    alias: str,
    max_length: int,
    max_new_tokens: int,
) -> str:
    prompt_ids = tokenizer(alias_text(decode_text, alias), add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= max_length:
        raise ValueError(f"Decode prompt length {len(prompt_ids)} exceeds {max_length}")
    device = next(wrapper.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    prompt_end = torch.tensor([len(prompt_ids)], device=device)
    output = wrapper.decode_from_state(
        state,
        input_ids=ids,
        attention_mask=attention,
        prompt_ends=prompt_end,
        use_cache=True,
        output_hidden_states=False,
    )
    past = output.past_key_values
    generated: list[int] = []
    token = int(output.logits[0, -1].argmax().item())
    for _ in range(max_new_tokens):
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if len(prompt_ids) + len(generated) >= max_length:
            break
        next_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        next_attention = torch.ones(
            1, len(prompt_ids) + len(generated), dtype=torch.long, device=device
        )
        output = wrapper.backbone(
            input_ids=next_ids,
            attention_mask=next_attention,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = output.past_key_values
        token = int(output.logits[0, -1].argmax().item())
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_batch(
    wrapper,
    tokenizer,
    states: list[torch.Tensor | None],
    decode_texts: list[str],
    alias: str,
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    """Greedily decode an equal-length, homogeneous state batch.

    Equal prompt lengths avoid padding-position changes.  Cold and populated
    states are never mixed, so each row follows the same forward path as
    ``generate_one``; batching changes only GPU packing, not token selection.
    """
    if not states or len(states) != len(decode_texts):
        raise ValueError("Generation batch must contain aligned states and prompts")
    cold = states[0] is None
    if any((state is None) != cold for state in states):
        raise ValueError("Generation batch mixed cold and populated states")
    prompt_ids = [
        tokenizer(alias_text(text, alias), add_special_tokens=False)["input_ids"]
        for text in decode_texts
    ]
    lengths = {len(ids) for ids in prompt_ids}
    if len(lengths) != 1:
        raise ValueError("Generation batch requires equal prompt lengths")
    prompt_length = lengths.pop()
    if prompt_length >= max_length:
        raise ValueError(
            f"Decode prompt length {prompt_length} exceeds {max_length}"
        )
    device = next(wrapper.parameters()).device
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    prompt_ends = torch.full(
        (len(states),), prompt_length, dtype=torch.long, device=device
    )
    state_batch = None if cold else torch.stack(states).to(device)
    output = wrapper.decode_from_state(
        state_batch,
        input_ids=ids,
        attention_mask=attention,
        prompt_ends=prompt_ends,
        use_cache=True,
        output_hidden_states=False,
    )
    past = output.past_key_values
    tokens = output.logits[:, -1].argmax(dim=-1)
    generated: list[list[int]] = [[] for _ in states]
    finished = [False] * len(states)
    for step in range(max_new_tokens):
        for index, token_tensor in enumerate(tokens):
            if finished[index]:
                continue
            token = int(token_tensor.item())
            if token == tokenizer.eos_token_id:
                finished[index] = True
                continue
            generated[index].append(token)
            if prompt_length + len(generated[index]) >= max_length:
                finished[index] = True
        if all(finished):
            break
        next_ids = tokens.unsqueeze(1)
        if any(finished):
            next_ids = next_ids.clone()
            for index, done in enumerate(finished):
                if done:
                    next_ids[index, 0] = tokenizer.eos_token_id
        next_attention = torch.ones(
            len(states),
            prompt_length + step + 1,
            dtype=torch.long,
            device=device,
        )
        output = wrapper.backbone(
            input_ids=next_ids,
            attention_mask=next_attention,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = output.past_key_values
        tokens = output.logits[:, -1].argmax(dim=-1)
    return [
        tokenizer.decode(tokens, skip_special_tokens=True).strip()
        for tokens in generated
    ]


@torch.inference_mode()
def generation_records(
    wrapper,
    tokenizer,
    rows,
    states: list[torch.Tensor | None],
    alias: str,
    max_length: int,
    max_new_tokens: int,
    batch_size: int = 1,
):
    if batch_size <= 0:
        raise ValueError("Generation batch size must be positive")
    records: list[dict[str, Any] | None] = [None] * len(rows)
    groups: dict[tuple[bool, int], list[int]] = {}
    texts = [closed_decode_text(row) for row in rows]
    for index, (text, state) in enumerate(zip(texts, states, strict=True)):
        prompt_length = len(
            tokenizer(alias_text(text, alias), add_special_tokens=False)["input_ids"]
        )
        groups.setdefault((state is None, prompt_length), []).append(index)
    completed = 0
    for indices in groups.values():
      for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        outputs = generate_batch(
            wrapper,
            tokenizer,
            [states[index] for index in selected],
            [texts[index] for index in selected],
            alias,
            max_length,
            max_new_tokens,
        )
        for index, output in zip(selected, outputs, strict=True):
            row = rows[index]
            metadata = evaluation_metadata(row)
            records[index] = {
                "example_id": row["example_id"],
                "task": row["task"],
                "source": metadata["source"],
                "protocol": metadata["protocol"],
                "delta": metadata["delta"],
                "target": row["target"],
                "raw_output": output,
            }
        completed += len(selected)
        if completed == len(rows) or completed == len(selected) or completed % 10 < len(selected):
            print(json.dumps({"generated": completed, "total": len(rows)}), flush=True)
    if any(record is None for record in records):
        raise AssertionError("Generation batching left an incomplete record")
    return [record for record in records if record is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--recurrent-state", type=Path)
    parser.add_argument(
        "--state-architecture",
        choices=("scalar", "layerwise"),
        default="scalar",
    )
    parser.add_argument("--decode-adapter")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-group", type=int)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--state-batch-size", type=int, default=8)
    parser.add_argument("--candidate-row-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-unredacted-predictions", action="store_true")
    parser.add_argument(
        "--t3-natural-schema",
        action="store_true",
        help="Request and reference label/diagnosis/evidence output for T3 rows.",
    )
    args = parser.parse_args()

    rows = []
    with args.eval_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if args.tasks:
        allowed = set(args.tasks)
        rows = [row for row in rows if row["task"] in allowed]
    if args.sources:
        allowed = set(args.sources)
        rows = [
            row for row in rows if evaluation_metadata(row)["source"] in allowed
        ]
    if args.t3_natural_schema:
        from state8_natural_curriculum_view import (
            apply_state8_natural_curriculum_view,
        )

        apply_state8_natural_curriculum_view(rows)
    unsharded_rows = len(rows)
    rows = evaluation_shard(rows, args.shard_index, args.num_shards)
    rows = sample_rows(rows, args.max_samples_per_group, args.seed)
    classification = [row for row in rows if row["task"] in BINARY_TASKS]
    generation = [row for row in rows if row["task"] in GENERATION_TASKS]
    tokenizer, wrapper = load_wrapper(
        args.model,
        args.adapter,
        args.recurrent_state,
        args.state_token_alias,
        args.state_architecture,
    )
    all_states = recurrent_states_batched(
        wrapper,
        tokenizer,
        rows,
        wrapper.state_token_rendered_alias,
        args.max_length,
        args.state_batch_size,
    )
    state_by_id = {
        str(row["example_id"]): state for row, state in zip(rows, all_states, strict=True)
    }
    decode_adapter = args.decode_adapter or args.adapter
    active_decode_adapter = activate_decode_adapter(
        wrapper, args.adapter, decode_adapter
    )
    classification_states = [state_by_id[str(row["example_id"])] for row in classification]
    generation_states = [state_by_id[str(row["example_id"])] for row in generation]
    records = candidate_records(
        wrapper,
        tokenizer,
        classification,
        classification_states,
        wrapper.state_token_rendered_alias,
        args.max_length,
        args.candidate_row_batch_size,
    )
    records.extend(
        generation_records(
            wrapper,
            tokenizer,
            generation,
            generation_states,
            wrapper.state_token_rendered_alias,
            args.max_length,
            args.max_new_tokens,
            args.generation_batch_size,
        )
    )
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "decode_adapter": decode_adapter,
        "active_decode_adapter": active_decode_adapter,
        "state_architecture": args.state_architecture,
        "state_shape": (
            [wrapper.hidden_size]
            if args.state_architecture == "scalar"
            else (
                [wrapper.layer_count, wrapper.hidden_size]
                if wrapper.state_token_count == 1
                else [wrapper.layer_count, wrapper.state_token_count, wrapper.hidden_size]
            )
        ),
        "recurrent_state": (
            str(args.recurrent_state) if args.recurrent_state is not None else None
        ),
        "artifact_sha256": {
            "adapter_model": artifact_sha256(
                Path(args.adapter) / "adapter_model.safetensors"
            ),
            "recurrent_state": (
                artifact_sha256(args.recurrent_state)
                if args.recurrent_state is not None
                else None
            ),
        },
        "eval_jsonl": str(args.eval_jsonl),
        "information_route": "sequential_updates_then_continuous_state_plus_current_query",
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "decode_time_current_query_kv_cache": True,
        "sampled_rows": len(rows),
        "unsharded_filtered_rows": unsharded_rows,
        "evaluation_shard": (
            None
            if args.shard_index is None
            else {"index": args.shard_index, "count": args.num_shards}
        ),
        "metrics_by_group": summarize(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    write_jsonl(args.output_dir / "predictions_redacted.jsonl", redacted_records(records))
    if args.save_unredacted_predictions:
        write_jsonl(args.output_dir / "predictions.jsonl", records)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
