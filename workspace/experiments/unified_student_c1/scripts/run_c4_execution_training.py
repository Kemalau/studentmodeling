"""Two matched, bounded QLoRA continuation arms with resumable state and eval."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from peft import PeftModel,prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed,get_linear_schedule_with_warmup
from c4_execution_data import rows,prompt,INSTRUCTION,STATE,digest_json
from c4_execution_model import encode,exchange,capture,generate
from c4_freegen_protocol import sha

def atomic(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def log(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj)+'\n')
    print(json.dumps(obj),flush=True)

@torch.no_grad()
def validation_loss(model,tok,m,arm,device):
    model.eval();sums={};counts={}
    for r in rows(m,'valid'):
        b,pos=encode(tok,r,arm,m['training']['max_length']);b={k:v.to(device) for k,v in b.items()}
        with exchange(model,pos,enabled=arm=='B' and r['mode']=='history'):
            loss=float(model(**b,use_cache=False).loss)
        n=int((b['labels']!=-100).sum());sums[r['mode']]=sums.get(r['mode'],0)+loss*n;counts[r['mode']]=counts.get(r['mode'],0)+n
    model.train()
    return {k:dict(loss=sums[k]/counts[k],tokens=counts[k]) for k in sums}

def evaluate(model,tok,m,out,adapter_hash,manifest_hash,label):
    out.mkdir(parents=True,exist_ok=True);model.eval()
    fp=dict(label=label,adapter_sha256=adapter_hash,manifest_sha256=manifest_hash,protocol='execution_v1_corrected_anchor')
    if (out/'lineage.json').exists():assert json.loads((out/'lineage.json').read_text())==fp
    atomic(out/'lineage.json',fp)
    records=[]
    for gi,g in enumerate(m['splits']['test']):
        path=out/(g['id'].replace(':','_')+'.json')
        if path.exists():
            saved=json.loads(path.read_text());assert saved['fingerprint']==fp
            records.extend(saved['records']);continue
        states={role:capture(model,tok,INSTRUCTION+h+STATE) for role,h in g['histories'].items()}
        hashes={role:hashlib.sha256(b''.join(x.float().cpu().numpy().tobytes() for x in vs)).hexdigest() for role,vs in states.items()}
        rs=[]
        for q in g['queries']:
            for cond in m['evaluation']['conditions']:
                source={'swap':'donor','noop':'recipient','random':'random','same_donor':'same_donor',
                        'menu_swap':'donor','menu_random':'random'}.get(cond)
                menu=None
                if cond.startswith('menu_'):
                    menu=sorted(set([q['targets']['donor'],q['targets']['recipient'],q['correct']]))
                    random.Random(g['id']+str(q['query_idx'])).shuffle(menu)
                p=prompt(g['histories']['donor' if cond=='donor_native' else 'recipient'],q,
                    g['rules']['donor'] if cond=='oracle' else None,menu)
                y=generate(model,tok,p,states[source] if source else None,m['evaluation']['max_new_tokens'])
                if y['stop']=='token_limit':y['parsed']=None;y['parse_type']='truncated'
                rs.append(dict(group_id=g['id'],pair_id=g['pair_id'],query_idx=q['query_idx'],condition=cond,
                    prompt_sha256=hashlib.sha256(p.encode()).hexdigest(),targets=q['targets'],correct=q['correct'],**y))
        atomic(path,dict(fingerprint=fp,state_hashes=hashes,records=rs));records.extend(rs)
        print(f'[eval] {label} group {gi+1}/{len(m["splits"]["test"])}',flush=True)
    assert len(records)==len(m['splits']['test'])*3*len(m['evaluation']['conditions'])
    atomic(out/'records.json',records)
    summary={c:dict(n=sum(r['condition']==c for r in records),
        donor_hits=sum(r['condition']==c and r['parsed']==r['targets']['donor'] for r in records),
        recipient_hits=sum(r['condition']==c and r['parsed']==r['targets']['recipient'] for r in records),
        invalid=sum(r['condition']==c and r['parsed'] is None for r in records)) for c in m['evaluation']['conditions']}
    atomic(out/'summary.json',summary)
    return summary

def main():
    p=argparse.ArgumentParser()
    for name in ('model','adapter','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=('A','B'),required=True)
    p.add_argument('--diagnostic',action='store_true',help='Only one backward, no optimizer update or scientific eval')
    a=p.parse_args();m=json.loads(a.manifest.read_text());cfg=m['training'];set_seed(cfg['seed'])
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    init_hash=sha(a.adapter/'adapter_model.safetensors');assert init_hash==m['init_adapter_sha256']
    sources={str(f.name):sha(f) for f in Path(__file__).parent.glob('c4_execution_*.py')}
    sources[Path(__file__).name]=sha(__file__)
    fp=dict(arm=a.arm,manifest_sha256=sha(a.manifest),initial_adapter_sha256=init_hash,training=cfg,code=sources)
    if (out/'lineage.json').exists():assert json.loads((out/'lineage.json').read_text())==fp
    atomic(out/'lineage.json',fp)
    tok=AutoTokenizer.from_pretrained(a.model);tok.pad_token=tok.eos_token
    assert sha(a.model/'tokenizer.json')=='aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=quant,torch_dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='eager')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    checkpoints=sorted([d for d in out.glob('checkpoint-*') if (d/'complete.json').exists()],key=lambda d:int(d.name.split('-')[-1]))
    last=checkpoints[-1] if checkpoints and not a.diagnostic else None
    model=PeftModel.from_pretrained(model,str(last or a.adapter),is_trainable=True)
    model.enable_input_require_grads();model.config.use_cache=False
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    for c in model.peft_config.values():
        assert c.r==8;c.lora_dropout=0.
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,cfg['warmup_steps'],cfg['steps'])
    start=0
    if last:
        ck=torch.load(last/'training.pt',map_location='cpu',weights_only=False)
        assert ck['fingerprint']==fp;start=ck['step'];optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    train=rows(m,'train');encoded=[encode(tok,r,a.arm,cfg['max_length']) for r in train]
    assert len(encoded)==cfg['steps']*cfg['accumulation']
    device=model.get_input_embeddings().weight.device
    atomic(out/'exposure.json',dict(rows=len(train),supervised_tokens=sum(int((b['labels']!=-100).sum()) for b,pos in encoded),
        target_order_sha256=digest_json([r['target'] for r in train]),max_sequence=max(b['input_ids'].shape[1] for b,pos in encoded)))
    # Baseline and final use exactly this decoder/mask; old numbers are not baseline.
    if a.arm=='A' and not a.diagnostic:
        if start: # Baseline must have completed before first optimizer update.
            assert (out/'baseline/summary.json').exists()
        else:evaluate(model,tok,m,out/'baseline',init_hash,sha(a.manifest),'baseline')
    model.train();optimizer.zero_grad(set_to_none=True)
    start_time=time.monotonic();total_tokens=0
    for step in range(120 if a.diagnostic else start,cfg['steps']):
        values=[]
        for j in range(cfg['accumulation']):
            index=step*cfg['accumulation']+j;b,pos=encoded[index];r=train[index]
            b={k:v.to(device) for k,v in b.items()}
            diagnostic_inputs=[]
            if a.diagnostic:
                def save_input(module,inputs,output):
                    output.retain_grad();diagnostic_inputs.append(output)
                handle=model.get_input_embeddings().register_forward_hook(save_input)
            with exchange(model,pos,enabled=a.arm=='B' and r['mode']=='history'):
                loss=model(**b,use_cache=False).loss
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step} {index}')
                (loss/cfg['accumulation']).backward()
            values.append(float(loss.detach()));total_tokens+=int((b['labels']!=-100).sum())
            if a.diagnostic:
                handle.remove()
                norm=sum(float(p.grad.float().square().sum()) for p in params if p.grad is not None)**.5
                assert norm>0
                source_grad=float(diagnostic_inputs[0].grad[0].float().norm())
                if a.arm=='B':assert source_grad>0,'Donor source must receive actual 32B gradient'
                atomic(out/'diagnostic.json',dict(loss=values[0],gradient_norm=norm,
                    source_embedding_gradient_norm=source_grad,
                    peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-start_time))
                return
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.0))
        if not np.isfinite(norm):raise RuntimeError(f'Nonfinite gradient at {step}')
        optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        log(out/'train.jsonl',dict(step=step+1,phase=train[step*cfg['accumulation']]['mode'],loss=sum(values)/len(values),
            grad_norm=norm,lr=scheduler.get_last_lr()[0],elapsed=time.monotonic()-start_time,
            supervised_tokens_this_invocation=total_tokens,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))
        if (step+1)%40==0:
            d=out/f'checkpoint-{step+1}';d.mkdir(exist_ok=True)
            model.save_pretrained(d,save_embedding_layers=False)
            torch.save(dict(fingerprint=fp,step=step+1,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),d/'training.pt')
            atomic(d/'complete.json',dict(step=step+1,adapter_sha256=sha(d/'adapter_model.safetensors')))
        if step+1 in (120,240):
            atomic(out/f'validation-{step+1}.json',validation_loss(model,tok,m,a.arm,device))
    if not (out/'validation-240.json').exists():
        atomic(out/'validation-240.json',validation_loss(model,tok,m,a.arm,device))
    final=out/'final_adapter';model.save_pretrained(final,save_embedding_layers=False)
    evaluate(model,tok,m,out/'final',sha(final/'adapter_model.safetensors'),sha(a.manifest),a.arm)
    atomic(out/'complete.json',dict(status='complete',arm=a.arm,steps=cfg['steps'],final_adapter_sha256=sha(final/'adapter_model.safetensors')))

if __name__=='__main__':main()
