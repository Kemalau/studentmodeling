#!/usr/bin/env python3
"""Clone a state-token embedding row and verify zero-training equivalence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_c1_lora import (
    BottleneckCollator,
    STATE_TOKEN,
    add_special_state_token,
    encode_example,
    read_jsonl,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(model_path: str, adapter_path: str):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def embedding_rows(model, source_id: int, target_id: int) -> dict[str, torch.Tensor]:
    input_weight = model.get_input_embeddings().weight
    output = model.get_output_embeddings()
    if target_id >= input_weight.shape[0]:
        raise ValueError(
            f"Target token id {target_id} exceeds input embedding size {input_weight.shape[0]}; "
            "resize explicitly before cloning"
        )
    rows = {
        "input_source": input_weight[source_id].detach().cpu().contiguous(),
        "input_target_before": input_weight[target_id].detach().cpu().contiguous(),
    }
    if output is not None:
        if target_id >= output.weight.shape[0]:
            raise ValueError(f"Target token id {target_id} exceeds output embedding size {output.weight.shape[0]}")
        rows.update(
            {
                "output_source": output.weight[source_id].detach().cpu().contiguous(),
                "output_target_before": output.weight[target_id].detach().cpu().contiguous(),
            }
        )
    return rows


def apply_rows(model, source_id: int, target_id: int, saved: dict[str, torch.Tensor] | None = None) -> None:
    input_weight = model.get_input_embeddings().weight
    output = model.get_output_embeddings()
    with torch.no_grad():
        source_input = saved["input_source"] if saved is not None else input_weight[source_id]
        input_weight[target_id].copy_(source_input.to(input_weight.device, input_weight.dtype))
        if output is not None:
            source_output = (
                saved["output_source"] if saved is not None and "output_source" in saved else output.weight[source_id]
            )
            output.weight[target_id].copy_(source_output.to(output.weight.device, output.weight.dtype))


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


@torch.inference_mode()
def compare_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    source_token: str,
    max_length: int,
) -> list[dict[str, Any]]:
    collator = BottleneckCollator(tokenizer, style="late_fusion", mask_prob=1.0)
    results: list[dict[str, Any]] = []
    for row in rows:
        source = encode_example(
            tokenizer,
            row,
            max_length=max_length,
            supervise_eos=False,
            bottleneck_mask=True,
            state_token_alias=source_token,
        )
        cloned = encode_example(
            tokenizer,
            row,
            max_length=max_length,
            supervise_eos=False,
            bottleneck_mask=True,
            state_token_alias=None,
        )
        source_ids = list(source["input_ids"])
        cloned_ids = list(cloned["input_ids"])
        differing = [i for i, (a, b) in enumerate(zip(source_ids, cloned_ids, strict=True)) if a != b]
        if differing != [int(source["state_pos"])]:
            raise AssertionError(f"Expected only the state token id to differ, got positions {differing}")
        batch = collator([source, cloned])
        device = getattr(model, "device", next(model.parameters()).device)
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
            output_hidden_states=True,
        )
        hidden_errors = [max_error(layer[0], layer[1]) for layer in output.hidden_states]
        results.append(
            {
                "example_id": row.get("example_id"),
                "sequence_length": len(source_ids),
                "state_position": int(source["state_pos"]),
                "logit_max_abs_error": max_error(output.logits[0], output.logits[1]),
                "hidden_max_abs_error": max(hidden_errors),
                "hidden_layer_max_abs_errors": hidden_errors,
            }
        )
    return results


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Zero-training state-token clone equivalence",
        "",
        f"- Source token: `{payload['source_token']}` (id {payload['source_id']})",
        f"- Target token: `{payload['target_token']}` (id {payload['target_id']})",
        f"- Samples: {payload['n_samples']}",
        f"- Maximum hidden-state absolute error: `{payload['max_hidden_abs_error']:.8g}`",
        f"- Maximum logit absolute error: `{payload['max_logit_abs_error']:.8g}`",
        f"- Reload maximum hidden-state absolute error: `{payload['reload_max_hidden_abs_error']:.8g}`",
        f"- Reload maximum logit absolute error: `{payload['reload_max_logit_abs_error']:.8g}`",
        f"- Passed: `{payload['passed']}`",
        "",
        "No optimizer step is used. The tokenizer changes by one registered token and the source input/output embedding rows are copied exactly.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer source; defaults to --adapter when the adapter bundles one.",
    )
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-token", default="<|vision_pad|>")
    parser.add_argument("--target-token", default=STATE_TOKEN)
    parser.add_argument("--task", default="malrule_next_response")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer_source = args.tokenizer or str(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    source_ids = tokenizer(args.source_token, add_special_tokens=False)["input_ids"]
    if len(source_ids) != 1:
        raise ValueError(f"Source token must be one id, got {source_ids}")
    added = add_special_state_token(tokenizer, args.target_token)
    target_ids = tokenizer(args.target_token, add_special_tokens=False)["input_ids"]
    if len(target_ids) != 1:
        raise ValueError(f"Target token must be one id, got {target_ids}")
    source_id, target_id = source_ids[0], target_ids[0]
    if source_id == target_id:
        raise ValueError("Source and target token ids must differ")

    selected = read_jsonl(args.jsonl, limit=args.samples, task_filter=args.task)
    if len(selected) != args.samples:
        raise ValueError(f"Requested {args.samples} samples but found {len(selected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    model = load_model(args.model, str(args.adapter))
    saved_rows = embedding_rows(model, source_id, target_id)
    before_input_error = max_error(saved_rows["input_source"], saved_rows["input_target_before"])
    before_output_error = (
        max_error(saved_rows["output_source"], saved_rows["output_target_before"])
        if "output_source" in saved_rows
        else None
    )
    apply_rows(model, source_id, target_id)
    after_rows = embedding_rows(model, source_id, target_id)
    input_row_error = max_error(after_rows["input_source"], after_rows["input_target_before"])
    output_row_error = (
        max_error(after_rows["output_source"], after_rows["output_target_before"])
        if "output_source" in after_rows
        else 0.0
    )
    save_file(
        {key: value for key, value in saved_rows.items() if key in {"input_source", "output_source"}},
        str(args.output_dir / "state_token_rows.safetensors"),
    )
    first_results = compare_rows(model, tokenizer, selected, args.source_token, args.max_length)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded_tokenizer = AutoTokenizer.from_pretrained(args.output_dir, trust_remote_code=True)
    if reloaded_tokenizer(args.target_token, add_special_tokens=False)["input_ids"] != [target_id]:
        raise AssertionError("Reloaded tokenizer changed the target token id")
    reloaded = load_model(args.model, str(args.adapter))
    row_payload = load_file(str(args.output_dir / "state_token_rows.safetensors"), device="cpu")
    apply_rows(reloaded, source_id, target_id, row_payload)
    reload_results = compare_rows(reloaded, reloaded_tokenizer, selected[:1], args.source_token, args.max_length)

    max_hidden = max(result["hidden_max_abs_error"] for result in first_results)
    max_logits = max(result["logit_max_abs_error"] for result in first_results)
    reload_max_hidden = max(result["hidden_max_abs_error"] for result in reload_results)
    reload_max_logits = max(result["logit_max_abs_error"] for result in reload_results)
    passed = max(input_row_error, output_row_error, max_hidden, max_logits, reload_max_hidden, reload_max_logits) <= args.tolerance
    adapter_model = args.adapter / "adapter_model.safetensors"
    payload = {
        "model": args.model,
        "adapter": str(args.adapter),
        "tokenizer_source": tokenizer_source,
        "adapter_sha256": sha256(adapter_model) if adapter_model.exists() else None,
        "source_token": args.source_token,
        "source_id": source_id,
        "target_token": args.target_token,
        "target_id": target_id,
        "target_added": added,
        "n_samples": len(first_results),
        "before_input_row_max_abs_error": before_input_error,
        "before_output_row_max_abs_error": before_output_error,
        "after_input_row_max_abs_error": input_row_error,
        "after_output_row_max_abs_error": output_row_error,
        "max_hidden_abs_error": max_hidden,
        "max_logit_abs_error": max_logits,
        "reload_max_hidden_abs_error": reload_max_hidden,
        "reload_max_logit_abs_error": reload_max_logits,
        "tolerance": args.tolerance,
        "passed": passed,
        "samples": first_results,
        "reload_samples": reload_results,
    }
    (args.output_dir / "equivalence.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key not in {"samples", "reload_samples"}}, indent=2))
    if not passed:
        raise SystemExit("Clone equivalence failed")


if __name__ == "__main__":
    main()
