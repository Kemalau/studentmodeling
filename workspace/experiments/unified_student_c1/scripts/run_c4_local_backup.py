"""Local backup only after exact output/state parity against completed shard 0.

Frozen evaluator/manifest are unchanged. Local output is separate; no remote
job is canceled here. Abort on parity failure or insufficient local headroom.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers, peft, bitsandbytes, accelerate, safetensors
from c4_freegen_protocol import sha
from c4_rule_followup import prompt
from evaluate_c4_format_diagnostic import atomic_json, generated
import evaluate_c4_rule_followup as evaluator
from state_swap_misconception_smoke import capture_state_vectors,load_model,locate_bottleneck_positions

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/c4_rule_followup_local_backup_20260905"
MODEL=Path("/home/v-yuejiali/studentmodeling/models/Qwen3-32B")
ADAPTER=Path("/mnt/codex-tmp/c4-followup-adapter/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter")
MANIFEST=ROOT/"manifests/c4_rule_followup_v2_seed42_20260905.json"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m=json.loads(MANIFEST.read_text())
    remote=ROOT/"results/downloads/c4-rule-followup-seed42-v2-20260905/c4-rule-followup-seed42-shard0/rule_followup"
    ref=json.loads((remote/"menu_000.json").read_text())
    assert sha(ADAPTER/"adapter_model.safetensors")==m["adapter_sha256"]
    assert sha(MODEL/"tokenizer.json")==ref["fingerprint"]["tokenizer_sha256"]
    for name,digest in ref["fingerprint"]["code"].items(): assert sha(ROOT/"scripts"/name)==digest
    free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    assert free>65000,f"Local free VRAM no longer sufficient: {free} MiB"
    versions={x.__name__:x.__version__ for x in (torch,transformers,peft,bitsandbytes,accelerate,safetensors)}
    base={"pid":os.getpid(),"versions":versions,"manifest_sha256":sha(MANIFEST),
          "started":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"local_gpu_initial_free_MiB":free,
          "scope":"Backup shard 2, not primary until explicit runtime provenance/parity audit; remote retained"}
    atomic_json(OUT/"status.json",dict(base,status="loading"))
    torch.manual_seed(42)
    tok,model=load_model(str(MODEL),str(ADAPTER),True,"<|vision_pad|>")
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
    group=m["menu_groups"][0]
    states={}
    for role,c in group["contexts"].items():
        states[role]=capture_state_vectors(model,tok,{"prompt":c["seg_instruction"]+c["seg_history"]+"<STUDENT_STATE>"},list(range(64)),1536,True,"standard","<|vision_pad|>")
    hashes={role:hashlib.sha256(b"".join(v[i].contiguous().numpy().tobytes() for i in range(64))).hexdigest() for role,v in states.items()}
    state_equal=hashes==ref["state_hashes"]
    sources={"none":None,"noop":states["recipient"],"swap":states["donor"],
             "same_donor_rule":states["same_donor_rule"],"same_recipient_rule":states["samemal"],"random_real":states["random_real"]}
    atomic_json(OUT/"status.json",dict(base,status="parity",state_equal=state_equal))
    checks=[];start=time.monotonic()
    # All 24 nonblank menu endpoints for one donor; not chosen from outcomes.
    for r in ref["records"]:
        _,_,qi,order,c=r["key"]
        if qi<0: continue
        ids,h0,sp=locate_bottleneck_positions(tok,prompt(m,group,qi,order,c),"<|vision_pad|>")
        actual=generated(model,tok,ids,h0,sp,sources[c],eos,48)
        checks.append({"key":r["key"],"tokens_equal":actual["tokens"]==r["tokens"],
                       "local":actual,"remote":{k:r[k] for k in actual}})
        print("[local-parity]",r["key"],checks[-1]["tokens_equal"],flush=True)
    oracle=json.loads((remote/"oracle_000.json").read_text())
    og=m["oracle_groups"][0]
    for r in oracle["records"]:
        _,_,qi,order,c=r["key"]
        ids,h0,sp=locate_bottleneck_positions(tok,prompt(m,og,qi,order,c),"<|vision_pad|>")
        actual=generated(model,tok,ids,h0,sp,None,eos,48)
        checks.append({"key":r["key"],"tokens_equal":actual["tokens"]==r["tokens"],
                       "local":actual,"remote":{k:r[k] for k in actual}})
    elapsed=time.monotonic()-start
    passed=state_equal and all(c["tokens_equal"] for c in checks)
    atomic_json(OUT/"parity.json",dict(base,state_equal=state_equal,state_hashes=hashes,
                remote_state_hashes=ref["state_hashes"],checks=checks,passed=passed,
                seconds=elapsed,estimated_shard_generation_minutes=elapsed/len(checks)*369/60,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
    if not passed:
        atomic_json(OUT/"status.json",dict(base,status="parity_failed",outputs_matched=sum(c["tokens_equal"] for c in checks),n=len(checks)))
        print("PARITY FAILED: no shard inference or primary merge",flush=True)
        return
    # Reuse the verified loaded model; the frozen evaluator resets seed itself.
    evaluator.load_model=lambda *a,**kw:(tok,model)
    sys.argv=["evaluate_c4_rule_followup.py","--model",str(MODEL),"--adapter",str(ADAPTER),
              "--manifest",str(MANIFEST),"--shard","2","--shards","3","--output-dir",str(OUT/"rule_followup")]
    atomic_json(OUT/"status.json",dict(base,status="running_shard2",parity_passed=True))
    evaluator.main()
    atomic_json(OUT/"status.json",dict(base,status="complete",parity_passed=True,
                finished=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))


if __name__=="__main__":
    try: main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/"status.json",{"status":"failed","pid":os.getpid(),"error":str(exc)})
        raise
