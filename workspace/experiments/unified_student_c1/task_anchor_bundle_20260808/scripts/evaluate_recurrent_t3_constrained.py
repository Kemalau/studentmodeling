#!/usr/bin/env python3
"""Diagnostic T3 decoding with an XML grammar over a frozen recurrent state.

This does not replace the frozen formal evaluator.  It fixes only the literal
opening/closing ``misconception`` tags while leaving every semantic content
token to greedy model decoding.  The result separates output-grammar failure
from loss of misconception information in the closed student state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_formal_protocols import sample_rows
from evaluate_recurrent_formal_protocols import (
    activate_decode_adapter,
    closed_decode_text,
    evaluation_metadata,
    load_wrapper,
    recurrent_states_batched,
)
from t3_judge_utils import redacted_records, summarize_t3, write_jsonl
from train_recurrent_student_diagnostic import alias_text


OPEN_TAG = "<misconception>"
CLOSE_TAG = "</misconception>"


@torch.inference_mode()
def constrained_generation(
    wrapper,
    tokenizer,
    state: torch.Tensor | None,
    decode_text: str,
    alias: str,
    max_length: int,
    max_content_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Generate semantic content inside a fixed XML envelope."""

    prompt_text = alias_text(decode_text, alias) + OPEN_TAG
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= max_length:
        raise ValueError(f"Constrained T3 prompt length {len(prompt_ids)} exceeds {max_length}")
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
    stop_reason = "max_content_tokens"
    for _ in range(max_content_tokens):
        if token == tokenizer.eos_token_id:
            stop_reason = "eos"
            break
        generated.append(token)
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if CLOSE_TAG.casefold() in decoded.casefold():
            stop_reason = "model_close_tag"
            break
        if len(prompt_ids) + len(generated) >= max_length:
            stop_reason = "max_length"
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
    content = tokenizer.decode(generated, skip_special_tokens=True).strip()
    close_index = content.casefold().find(CLOSE_TAG.casefold())
    model_closed = close_index >= 0
    if model_closed:
        content = content[:close_index].rstrip()
    raw_output = OPEN_TAG + content + CLOSE_TAG
    return raw_output, {
        "model_emitted_close_tag": model_closed,
        "stop_reason": stop_reason,
        "content_token_count": len(generated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--decode-adapter")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--max-samples-per-group", type=int, default=25)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-content-tokens", type=int, default=112)
    parser.add_argument("--state-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.eval_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("task") == "diagnose_misconception"]
    rows = sample_rows(rows, args.max_samples_per_group, args.seed)
    if not rows:
        raise ValueError("No T3 rows selected")
    tokenizer, wrapper = load_wrapper(
        args.model,
        args.adapter,
        args.recurrent_state,
        args.state_token_alias,
        "layerwise",
    )
    states = recurrent_states_batched(
        wrapper,
        tokenizer,
        rows,
        args.state_token_alias,
        args.max_length,
        args.state_batch_size,
    )
    decode_adapter = args.decode_adapter or args.adapter
    active_decode_adapter = activate_decode_adapter(wrapper, args.adapter, decode_adapter)
    records: list[dict[str, Any]] = []
    for index, (row, state_cpu) in enumerate(zip(rows, states, strict=True), start=1):
        state = (
            None
            if state_cpu is None
            else state_cpu.unsqueeze(0).to(next(wrapper.parameters()).device)
        )
        raw_output, diagnostic = constrained_generation(
            wrapper,
            tokenizer,
            state,
            closed_decode_text(row),
            args.state_token_alias,
            args.max_length,
            args.max_content_tokens,
        )
        metadata = evaluation_metadata(row)
        records.append(
            {
                "example_id": row["example_id"],
                "task": row["task"],
                "source": metadata["source"],
                "protocol": metadata["protocol"],
                "delta": metadata["delta"],
                "target": row["target"],
                "raw_output": raw_output,
                **diagnostic,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(json.dumps({"generated": index, "total": len(rows)}), flush=True)

    metrics = summarize_t3(records)
    model_close_rate = sum(bool(row["model_emitted_close_tag"]) for row in records) / len(records)
    result = {
        "stage": "checkpoint750_closed_state_T3_constrained_decode_diagnostic",
        "diagnostic_only": True,
        "frozen_formal_evaluator_unchanged": True,
        "model": args.model,
        "adapter": args.adapter,
        "decode_adapter": decode_adapter,
        "active_decode_adapter": active_decode_adapter,
        "recurrent_state": str(args.recurrent_state),
        "state_architecture": "layerwise",
        "state_shape": [wrapper.layer_count, wrapper.hidden_size],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "forced_open_tag": OPEN_TAG,
        "forced_close_tag": CLOSE_TAG,
        "semantic_content_is_model_generated": True,
        "sampled_rows": len(rows),
        "model_emitted_close_tag_rate": model_close_rate,
        "metrics_by_group": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    write_jsonl(args.output_dir / "predictions_redacted.jsonl", redacted_records(records))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
