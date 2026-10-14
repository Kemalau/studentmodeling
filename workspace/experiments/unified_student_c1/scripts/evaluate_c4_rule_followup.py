"""Frozen CF1 checkpoint: two inference-only follow-ups, resumable per donor."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from c4_freegen_protocol import sha
from c4_rule_followup import expected, group_keys, prompt, record_id
from evaluate_c4_format_diagnostic import atomic_json, generated
from state_swap_misconception_smoke import capture_state_vectors, get_decoder_layers, load_model, locate_bottleneck_positions


def main():
    ap = argparse.ArgumentParser()
    for name in ("model","adapter","manifest","output-dir"):
        ap.add_argument("--"+name,required=True)
    ap.add_argument("--shard",type=int,required=True)
    ap.add_argument("--shards",type=int,default=3)
    args = ap.parse_args()
    m = json.loads(Path(args.manifest).read_text())
    assert 0 <= args.shard < args.shards
    assert sha(Path(args.adapter)/"adapter_model.safetensors") == m["adapter_sha256"]
    tokenizer_path = Path(args.adapter)/"tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path(args.model)/"tokenizer.json"
    scripts = Path(__file__).resolve().parent
    fp = {"manifest_sha256":sha(args.manifest),"adapter_sha256":m["adapter_sha256"],
          "base_config_sha256":sha(Path(args.model)/"config.json"),
          "base_index_sha256":sha(Path(args.model)/"model.safetensors.index.json"),
          "tokenizer_sha256":sha(tokenizer_path),"shard":args.shard,"shards":args.shards,
          "code":{n:sha(scripts/n) for n in (Path(__file__).name,"c4_rule_followup.py","c4_freegen_protocol.py",
                   "evaluate_c4_format_diagnostic.py","evaluate_c4_freegen_novel.py","state_swap_misconception_smoke.py")}}
    assert fp["base_config_sha256"] == "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
    assert fp["base_index_sha256"] == "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
    assert fp["tokenizer_sha256"] == "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"lineage.json").exists():
        assert json.loads((out/"lineage.json").read_text()) == fp
    else:
        atomic_json(out/"lineage.json",fp)
    torch.manual_seed(42)
    tokenizer,model = load_model(args.model,args.adapter,True,"<|vision_pad|>")
    layers = list(range(len(get_decoder_layers(model))))
    assert len(layers) == 64
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos]) | {tokenizer.eos_token_id}
    cache, all_records = {},[]
    for arm,gs in (("oracle",m["oracle_groups"]),("menu",m["menu_groups"])):
        for g in gs:
            if g["group_idx"] % args.shards != args.shard:
                continue
            target = out/f"{arm}_{g['group_idx']:03d}.json"
            want = set(group_keys(arm,g))
            if target.exists():
                prev = json.loads(target.read_text())
                assert prev["fingerprint"] == fp
                assert len(prev["records"]) == len(want) and {tuple(r["key"]) for r in prev["records"]} == want
                all_records.extend(prev["records"])
                continue
            started = time.monotonic()
            states, state_hashes = {},{}
            if arm == "menu":
                for role,c in g["contexts"].items():
                    rid = c["example_id"]
                    if rid not in cache:
                        prefix = c["seg_instruction"]+c["seg_history"]+"<STUDENT_STATE>"
                        cache[rid] = capture_state_vectors(model,tokenizer,{"prompt":prefix},layers,1536,True,"standard","<|vision_pad|>")
                    states[role] = cache[rid]
                    state_hashes[role] = hashlib.sha256(b"".join(cache[rid][i].contiguous().numpy().tobytes() for i in layers)).hexdigest()
                sources = {"none":None,"noop":states["recipient"],"swap":states["donor"],
                           "same_donor_rule":states["same_donor_rule"],"same_recipient_rule":states["samemal"],"random_real":states["random_real"]}
            records = []
            for key in group_keys(arm,g):
                _,_,qi,order,c = key
                text = prompt(m,g,qi,order,c)
                digest = hashlib.sha256(text.encode()).hexdigest()
                assert digest == g["prompt_hashes"][record_id(key)]
                ids,h0,sp = locate_bottleneck_positions(tokenizer,text,"<|vision_pad|>")
                assert len(ids)+m["max_new_tokens"] <= 1536
                result = generated(model,tokenizer,ids,h0,sp,None if arm=="oracle" else sources[c],eos,m["max_new_tokens"])
                records.append({"key":key,"prompt_sha256":digest,"donor_id":g["ids"]["donor"],"state_hashes":state_hashes,**result})
                print(f"[rule-followup] {record_id(key)} parsed={result['parsed']!r}",flush=True)
            atomic_json(target,{"fingerprint":fp,"state_hashes":state_hashes,"records":records})
            all_records.extend(records)
            print(f"[rule-followup] finished {arm} group={g['group_idx']} seconds={time.monotonic()-started:.1f}",flush=True)
    want = {x for x in expected(m) if x[1] % args.shards == args.shard}
    assert len(all_records)==len(want) and {tuple(r["key"]) for r in all_records}==want
    tmp = out/"records.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r)+"\n" for r in all_records))
    tmp.replace(out/"records.jsonl")
    print(f"[rule-followup] DONE shard={args.shard} records={len(all_records)}",flush=True)


if __name__ == "__main__":
    main()
