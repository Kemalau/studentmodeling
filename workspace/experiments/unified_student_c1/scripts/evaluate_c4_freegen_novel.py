"""Greedy C4 generation with fixed per-donor trajectories and resumable shards."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from c4_freegen_protocol import CONDITIONS, parse_output, render_prompt, sha
from state_swap_misconception_smoke import (
    build_bottleneck_attention_mask, capture_state_vectors, get_decoder_layers,
    load_model, locate_bottleneck_positions,
)


def patch_hooks(model, state_pos, vectors):
    if vectors is None:
        return []
    handles = []
    for layer, vector in vectors.items():
        def hook(_module, _inputs, output, vec=vector):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            patched[:, state_pos, :] = vec.to(device=patched.device, dtype=patched.dtype)
            return (patched,) + output[1:] if isinstance(output, tuple) else patched
        handles.append(get_decoder_layers(model)[layer].register_forward_hook(hook))
    return handles


@torch.inference_mode()
def greedy_tokens(model, ids, history_start, state_pos, vectors, max_new_tokens, eos_ids,
                  tokenizer=None, cached=True):
    """Prefill with the original late-fusion mask; each answer token sees state.

    Only the prefill state position is patched for cached decoding. Its effect
    persists in downstream-layer K/V. A new cache is created for each condition
    and query, so no previous query or generated answer contaminates a state.
    """
    device = model.get_input_embeddings().weight.device
    prompt_end = len(ids)
    generated, past = [], None
    for step in range(max_new_tokens):
        incremental = cached and step > 0
        if incremental:
            tensor = torch.tensor([[generated[-1]]], device=device)
            mask = torch.zeros((1, 1, 1, prompt_end + step), dtype=torch.float32, device=device)
            mask[..., history_start:state_pos] = torch.finfo(torch.float32).min
            hooks = []
        else:
            tensor = torch.tensor([ids + generated], device=device)
            mask = build_bottleneck_attention_mask(
                torch.ones_like(tensor), torch.tensor([history_start], device=device),
                torch.tensor([state_pos], device=device), torch.tensor([prompt_end], device=device),
                style="late_fusion",
            )
            hooks = patch_hooks(model, state_pos, vectors)
        try:
            output = model(input_ids=tensor, attention_mask=mask, past_key_values=past,
                           use_cache=cached)
        finally:
            for handle in hooks:
                handle.remove()
        last_logits = output.logits[0, -1].float()
        token = int(last_logits.argmax().item())
        if cached:
            past = output.past_key_values
        del output
        if token in eos_ids:
            return generated, "eos"
        generated.append(token)
        if tokenizer is not None and "</predicted_response>" in tokenizer.decode(generated, skip_special_tokens=False):
            return generated, "closing_tag"
    return generated, "token_limit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    manifest = json.loads(args.manifest.read_text())
    assert sha(Path(args.adapter) / "adapter_model.safetensors") == manifest["adapter_sha256"]
    fingerprint = {"manifest_sha256": sha(args.manifest), "adapter_sha256": manifest["adapter_sha256"],
                   "evaluator_sha256": sha(__file__), "protocol_sha256": sha(Path(__file__).with_name("c4_freegen_protocol.py")),
                   "swap_helpers_sha256": sha(Path(__file__).with_name("state_swap_misconception_smoke.py")),
                   "base_config_sha256": sha(Path(args.model) / "config.json"),
                   "base_index_sha256": sha(Path(args.model) / "model.safetensors.index.json"),
                   "tokenizer_sha256": sha(Path(args.adapter) / "tokenizer.json") if (Path(args.adapter) / "tokenizer.json").exists() else sha(Path(args.model) / "tokenizer.json"),
                   "shard": args.shard, "shards": args.shards, "max_new_tokens": manifest["max_new_tokens"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.output_dir / "lineage.json"
    if receipt.exists():
        assert json.loads(receipt.read_text()) == fingerprint, "Resume lineage mismatch"
    else:
        receipt.write_text(json.dumps(fingerprint, indent=2) + "\n")
    torch.manual_seed(42)
    tokenizer, model = load_model(args.model, args.adapter, True, "<|vision_pad|>")
    layers = list(range(len(get_decoder_layers(model))))
    assert len(layers) == 64
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    state_cache = {}
    all_records = []
    groups = [g for g in manifest["groups"] if g["group_idx"] % args.shards == args.shard]
    for gi, group in enumerate(groups, 1):
        output_path = args.output_dir / f"group_{group['group_idx']:03d}.json"
        if output_path.exists():
            saved = json.loads(output_path.read_text())
            assert saved["fingerprint"] == fingerprint
            expected = {(q["query_idx"], c) for q in group["queries"] for c in CONDITIONS}
            got = [(r["query_idx"], r["condition"]) for r in saved["records"]]
            assert len(got) == len(set(got)) and set(got) == expected
            all_records.extend(saved["records"])
            print(f"[freegen] resume group={group['group_idx']}", flush=True)
            continue
        started = time.monotonic()
        states = {}
        for kind, context in group["contexts"].items():
            row_id = context["example_id"]
            if row_id not in state_cache:
                # Capture only the history prefix through the state address;
                # no target query, candidates, labels or correct answer are read.
                capture_row = {"prompt": context["seg_instruction"] + context["seg_history"] + "<STUDENT_STATE>"}
                state_cache[row_id] = capture_state_vectors(model, tokenizer, capture_row, layers, 1536,
                                                           True, "standard", "<|vision_pad|>")
            states[kind] = state_cache[row_id]
        records = []
        for query in group["queries"]:
            prompt = render_prompt(group["contexts"]["recipient"], query["problem_text"])
            ids, h0, spos = locate_bottleneck_positions(tokenizer, prompt, "<|vision_pad|>")
            assert len(ids) + manifest["max_new_tokens"] <= 1536, "No silent truncation"
            conditions = {"none": None, "noop": states["recipient"], "swap": states["donor"],
                          "random_real": states["random_real"], "samemal": states["samemal"]}
            for condition, vectors in conditions.items():
                tokens, stop = greedy_tokens(model, ids, h0, spos, vectors, manifest["max_new_tokens"], eos_ids, tokenizer)
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                parsed, parse_type = parse_output(text)
                record = {"group_idx": group["group_idx"], "pair_idx": group["pair_idx"],
                          "query_idx": query["query_idx"], "condition": condition,
                          "donor_id": group["ids"]["donor"], "recipient_id": group["ids"]["recipient"],
                          "donor_rule": group["labels"]["donor"], "family": group["family"],
                          "random_rule_applicable": query["random_rule_target"] is not None,
                          "target_seen_in_finetuning_answers": query["target_seen_in_finetuning_answers"],
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "text": text, "tokens": tokens, "stop": stop, "parsed": parsed, "parse_type": parse_type,
                          "matches_A": parsed == query["donor_target"], "matches_B": parsed == query["recipient_target"],
                          "matches_correct": parsed == query["correct_target"],
                          "matches_random_rule": parsed is not None and parsed == query["random_rule_target"],
                          "donor_target": query["donor_target"], "recipient_target": query["recipient_target"],
                          "correct_target": query["correct_target"]}
                records.append(record)
                print(f"[freegen] group={group['group_idx']} query={query['query_idx']} condition={condition} parsed={parsed!r} A={record['matches_A']}", flush=True)
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint, "records": records}, indent=2) + "\n")
        os.replace(tmp, output_path)
        all_records.extend(records)
        print(f"[freegen] finished group={gi}/{len(groups)} seconds={time.monotonic()-started:.1f}", flush=True)
    target = args.output_dir / "records.jsonl"
    target.write_text("".join(json.dumps(r) + "\n" for r in all_records))
    print(f"[freegen] DONE shard={args.shard} records={len(all_records)}", flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
