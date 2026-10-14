"""Frozen, resumable, inference-only history-length shards. No training."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from c4_history_diagnostic import COHORTS, CONDITIONS, validate
from c4_unseen_rules import INSTRUCTION, STATE, digest, prompt
from c4_execution_model import capture, generate
from c4_freegen_protocol import sha
from evaluate_c4_unseen_rules import atomic, open_generate
from state_swap_misconception_smoke import get_decoder_layers, locate_bottleneck_positions


def main():
    p=argparse.ArgumentParser()
    for name in ('model','adapter','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=('A','B'),required=True)
    p.add_argument('--cohort',choices=COHORTS,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--max-new-groups',type=int,default=0)
    a=p.parse_args();m=json.loads(a.manifest.read_text());validate(m)
    assert a.max_new_groups>=0
    assert sha(a.adapter/'adapter_model.safetensors')==m['checkpoints'][a.arm]['adapter_sha256']
    expected={'config.json':'97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb',
        'model.safetensors.index.json':'bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0',
        'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'}
    for name,value in expected.items():assert sha(a.model/name)==value
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.pad_token=tok.eos_token
    gs=[g for g in m['groups'] if g['cohort']==a.cohort]
    lengths=[]
    for g in gs:
        for q in g['queries']:
            for c in CONDITIONS:
                text=prompt(g,q,c);ids,_,_=locate_bottleneck_positions(tok,text,'<|vision_pad|>')
                assert len(ids)+m['max_new_tokens']<=m['max_length'],(g['id'],len(ids))
                lengths.append(len(ids))
    print(json.dumps(dict(stage='preflight_pass',arm=a.arm,cohort=a.cohort,
        groups=len(gs),generations=len(lengths),max_prompt_tokens=max(lengths))),flush=True)
    if a.preflight_only:return
    a.output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).parent
    names=('evaluate_c4_history_diagnostic.py','c4_history_diagnostic.py','evaluate_c4_unseen_rules.py',
        'c4_unseen_rules.py','c4_execution_model.py','c4_execution_data.py','c4_rule_followup.py',
        'c4_freegen_protocol.py','train_c1_lora.py','state_swap_misconception_smoke.py')
    fp=dict(arm=a.arm,cohort=a.cohort,manifest_sha256=sha(a.manifest),base=expected,
        adapter_sha256=m['checkpoints'][a.arm]['adapter_sha256'],code={n:sha(source/n) for n in names},
        precision='NF4/bfloat16 with kbit preparation; frozen rank8; greedy48')
    if (a.output/'lineage.json').exists():assert json.loads((a.output/'lineage.json').read_text())==fp
    atomic(a.output/'lineage.json',fp)
    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    assert free>=45000,('Insufficient headroom',free)
    set_seed(42);started=time.monotonic()
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
        quantization_config=quant,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(model,str(a.adapter),is_trainable=False)
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    assert all(c.r==8 for c in model.peft_config.values()) and len(get_decoder_layers(model))==64
    model.eval()
    runtime=dict(fingerprint=fp,initial_free_mib=free,load_seconds=time.monotonic()-started,
        versions={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','bitsandbytes','accelerate','safetensors')},
        gpu=torch.cuda.get_device_name(0),max_prompt_tokens=max(lengths),groups=[])
    records=[]
    for gi,g in enumerate(gs):
        path=a.output/(g['id'].replace(':','_')+'.json')
        if path.exists():
            previous=json.loads(path.read_text());assert previous['fingerprint']==fp
            assert len(previous['records'])==3*len(CONDITIONS)
            assert {(r['query_idx'],r['condition']) for r in previous['records']}=={(i,c) for i in range(3) for c in CONDITIONS}
            records.extend(previous['records']);continue
        start=time.monotonic()
        states={r:capture(model,tok,INSTRUCTION+g['histories'][r]+STATE) for r in ('recipient','donor','random')}
        hashes={r:hashlib.sha256(b''.join(v.float().cpu().numpy().tobytes() for v in vs)).hexdigest() for r,vs in states.items()}
        rs=[]
        for q in g['queries']:
            for c in CONDITIONS:
                text=prompt(g,q,c);role={'noop':'recipient','swap':'donor','random':'random'}.get(c)
                y=open_generate(model,tok,text,m['max_new_tokens']) if c=='open_history' else generate(model,tok,text,states[role] if role else None,m['max_new_tokens'])
                if y['stop']=='token_limit':y['parsed']=None;y['parse_type']='truncated'
                rs.append(dict(group_id=g['id'],original_group_id=g['original_group_id'],cohort=a.cohort,
                    pair_id=g['pair_id'],category=g['category'],query_idx=q['query_idx'],condition=c,
                    prompt_sha256=digest(text),targets=q['targets'],correct=q['correct'],**y))
        atomic(path,dict(fingerprint=fp,state_hashes=hashes,records=rs));records.extend(rs)
        runtime['groups'].append(dict(group_id=g['id'],seconds=time.monotonic()-start,records=len(rs)))
        runtime.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        atomic(a.output/'runtime.json',runtime)
        print(f'[history-diagnostic] {a.arm}/{a.cohort} {gi+1}/{len(gs)} seconds={time.monotonic()-start:.1f}',flush=True)
        if a.max_new_groups and len(runtime['groups'])>=a.max_new_groups and len(records)<m['n_generations_per_shard']:
            print('Bounded smoke done; not a complete shard',flush=True);return
    assert len(records)==m['n_generations_per_shard']
    atomic(a.output/'records.json',records)
    atomic(a.output/'complete.json',dict(status='complete',fingerprint=fp,records=len(records),records_sha256=sha(a.output/'records.json')))


if __name__=='__main__':main()
