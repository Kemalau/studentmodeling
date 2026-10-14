"""No training: fixed trajectories, four prompt formats, two readout modes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from c4_freegen_protocol import CONDITIONS, parse_output, sha
from c4_format_diagnostic import FORMATS, MODES, expected, key, render
from evaluate_c4_freegen_novel import greedy_tokens
from state_swap_misconception_smoke import capture_state_vectors, get_decoder_layers, load_model, locate_bottleneck_positions, score_row


def atomic_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def generated(model, tokenizer, ids, h0, spos, vectors, eos, limit, cached=True):
    tokens, stop = greedy_tokens(model, ids, h0, spos, vectors, limit, eos, tokenizer, cached=cached)
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    parsed, parse_type = parse_output(text)
    return {"tokens": tokens, "stop": stop, "text": text, "parsed": parsed, "parse_type": parse_type}


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "adapter", "manifest", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    manifest = json.loads(Path(args.manifest).read_text())
    assert sha(Path(args.adapter) / "adapter_model.safetensors") == manifest["adapter_sha256"]
    script_dir = Path(__file__).resolve().parent
    tokenizer_path = Path(args.adapter) / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path(args.model) / "tokenizer.json"
    fingerprint = {"manifest_sha256": sha(args.manifest), "adapter_sha256": manifest["adapter_sha256"],
                   "code": {name: sha(script_dir / name) for name in
                            (Path(__file__).name, "c4_format_diagnostic.py", "c4_freegen_protocol.py",
                             "evaluate_c4_freegen_novel.py", "state_swap_misconception_smoke.py")},
                   "base_config_sha256": sha(Path(args.model) / "config.json"),
                   "base_index_sha256": sha(Path(args.model) / "model.safetensors.index.json"),
                   "tokenizer_sha256": sha(tokenizer_path), "shard": args.shard, "shards": args.shards,
                   "max_new_tokens": manifest["max_new_tokens"]}
    # Match the previously audited experiment, not just the other new shards.
    assert fingerprint["base_config_sha256"] == "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
    assert fingerprint["base_index_sha256"] == "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
    assert fingerprint["tokenizer_sha256"] == "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lineage = out / "lineage.json"
    if lineage.exists():
        assert json.loads(lineage.read_text()) == fingerprint, "Resume lineage mismatch"
    else:
        atomic_json(lineage, fingerprint)
    torch.manual_seed(42)
    tokenizer, model = load_model(args.model, args.adapter, True, "<|vision_pad|>")
    layers = list(range(len(get_decoder_layers(model))))
    assert len(layers) == 64
    stop_ids = model.generation_config.eos_token_id
    eos = set(stop_ids if isinstance(stop_ids, list) else [stop_ids]) | {tokenizer.eos_token_id}
    cache, all_records = {}, []
    groups = [g for g in manifest["groups"] if g["group_idx"] % args.shards == args.shard]
    for gi, group in enumerate(groups, 1):
        group_path = out / f"group_{group['group_idx']:03d}.json"
        if group_path.exists():
            previous = json.loads(group_path.read_text())
            assert previous["fingerprint"] == fingerprint
            want = {k for k in expected(manifest, args.shard, args.shards) if k[0] == group["group_idx"]}
            assert len(previous["records"]) == len(want)
            assert {key(r) for r in previous["records"]} == want
            all_records.extend(previous["records"])
            print(f"[format] resume group={group['group_idx']}", flush=True)
            continue
        started = time.monotonic()
        states = {}
        for kind, context in group["contexts"].items():
            row_id = context["example_id"]
            if row_id not in cache:
                prefix = context["seg_instruction"] + context["seg_history"] + "<STUDENT_STATE>"
                cache[row_id] = capture_state_vectors(model, tokenizer, {"prompt": prefix}, layers, 1536,
                                                      True, "standard", "<|vision_pad|>")
            states[kind] = cache[row_id]
        state_hashes = {kind: hashlib.sha256(b"".join(v[i].contiguous().numpy().tobytes() for i in layers)).hexdigest()
                        for kind, v in states.items()}
        sources = {"none": None, "noop": states["recipient"], "swap": states["donor"],
                   "random_real": states["random_real"], "samemal": states["samemal"]}
        records, numerical = [], []
        for query in group["queries"]:
            for fmt in FORMATS:
                prompt = render(group, query, fmt)
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                assert prompt_hash == query["prompt_sha256_by_format"][fmt]
                ids, h0, spos = locate_bottleneck_positions(tokenizer, prompt, "<|vision_pad|>")
                assert len(ids) + manifest["max_new_tokens"] <= 1536, "No truncation allowed"
                row = {"prompt": prompt, "candidate_labels": query["candidate_labels"], "candidate_targets": query["candidate_targets"]}
                for condition, vectors in sources.items():
                    base = {"group_idx": group["group_idx"], "pair_idx": group["pair_idx"], "query_idx": query["query_idx"],
                            "format": fmt, "condition": condition, "donor_id": group["ids"]["donor"],
                            "recipient_id": group["ids"]["recipient"], "family": group["family"],
                            "donor_rule": group["labels"]["donor"], "prompt_sha256": prompt_hash,
                            "random_rule_applicable": query["random_rule_target"] is not None,
                            "target_seen_in_finetuning_answers": query["target_seen_in_finetuning_answers"],
                            "donor_target": query["donor_target"], "recipient_target": query["recipient_target"],
                            "correct_target": query["correct_target"]}
                    scores = score_row(model, tokenizer, row, layers, vectors, 1536, True, "late_fusion", "<|vision_pad|>")
                    generation = generated(model, tokenizer, ids, h0, spos, vectors, eos, manifest["max_new_tokens"])
                    scored = {**scores, "parsed": scores["pred"], "text": query["candidate_targets"][scores["pred"]], "parse_type": "candidate_score"}
                    for mode, result in (("candidate_score", scored), ("freegen", generation)):
                        record = dict(base, mode=mode, **result)
                        for flag, target in (("matches_A", "donor_target"), ("matches_B", "recipient_target"), ("matches_correct", "correct_target")):
                            record[flag] = record["parsed"] == query[target]
                        records.append(record)
                    print(f"[format] group={group['group_idx']} query={query['query_idx']} fmt={fmt} cond={condition} score={scores['pred']} generated={generation['parsed']}", flush=True)

                # Explicitly post-hoc diagnosis of the two previously observed no-op discrepancies.
                # These extra generations are not mixed into the 3,240 factorial endpoints.
                if fmt == "neither" and (group["group_idx"], query["query_idx"]) in {(2, 1), (22, 0)}:
                    full_state = capture_state_vectors(model, tokenizer, {"prompt": prompt}, layers, 1536,
                                                       True, "late_fusion", "<|vision_pad|>")
                    numerical.append({"group_idx": group["group_idx"], "query_idx": query["query_idx"],
                                      "prefix_full_state_max_abs_by_layer": {i: float((full_state[i] - states["recipient"][i]).abs().max()) for i in layers},
                                      "native_full_recompute": generated(model, tokenizer, ids, h0, spos, None, eos, 48, cached=False),
                                      "prefix_noop_full_recompute": generated(model, tokenizer, ids, h0, spos, states["recipient"], eos, 48, cached=False),
                                      "fullprompt_noop_cached": generated(model, tokenizer, ids, h0, spos, full_state, eos, 48)})
        atomic_json(group_path, {"fingerprint": fingerprint, "state_hashes": state_hashes, "records": records,
                                 "numerical_diagnostics": numerical})
        all_records.extend(records)
        print(f"[format] finished group={gi}/{len(groups)} seconds={time.monotonic()-started:.1f}", flush=True)
    assert len(all_records) == len(expected(manifest, args.shard, args.shards))
    assert {key(r) for r in all_records} == expected(manifest, args.shard, args.shards)
    temp = out / "records.jsonl.tmp"
    temp.write_text("".join(json.dumps(r) + "\n" for r in all_records))
    os.replace(temp, out / "records.jsonl")
    print(f"[format] DONE shard={args.shard} records={len(all_records)}", flush=True)


if __name__ == "__main__":
    main()
