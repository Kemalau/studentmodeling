"""Inference only: same frozen A/B precision/masks, plus open-history diagnostic."""
import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path
import time

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from c4_execution_model import capture, generate
from c4_freegen_protocol import parse_output, sha
from c4_unseen_rules import CONDITIONS, INSTRUCTION, STATE, digest, prompt
from state_swap_misconception_smoke import get_decoder_layers, locate_bottleneck_positions
from train_c1_lora import build_causal_4d_attention_mask


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)


@torch.no_grad()
def open_generate(model, tok, text, limit):
    """Same text/tokenization/decoding, ordinary causal attention to history."""
    ids, _, _ = locate_bottleneck_positions(tok,text,'<|vision_pad|>')
    device = model.get_input_embeddings().weight.device
    tokens = []; past = None; stop = 'token_limit'
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos]) | {tok.eos_token_id}
    for step in range(limit):
        t = torch.tensor([ids] if step==0 else [[tokens[-1]]],device=device)
        mask = (build_causal_4d_attention_mask(torch.ones_like(t)) if step==0
                else torch.zeros((1,1,1,len(ids)+step),device=device))
        out = model(input_ids=t,attention_mask=mask,past_key_values=past,use_cache=True)
        past = out.past_key_values; token = int(out.logits[0,-1].argmax()); del out
        if token in eos: stop = 'eos'; break
        tokens.append(token)
        if '</predicted_response>' in tok.decode(tokens,skip_special_tokens=False):
            stop = 'closing_tag'; break
    text = tok.decode(tokens,skip_special_tokens=False)
    parsed, kind = parse_output(text)
    if stop=='token_limit': parsed, kind = None, 'truncated'
    return dict(tokens=tokens,text=text,parsed=parsed,parse_type=kind,stop=stop,prompt_tokens=len(ids))


def main():
    p = argparse.ArgumentParser()
    for name in ('model','adapter','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=('A','B'),required=True)
    p.add_argument('--max-new-groups',type=int,default=0,
                   help='Stop after this many new groups; zero evaluates all. Resume uses the same fingerprint.')
    p.add_argument('--runtime-report',type=Path)
    p.add_argument('--min-free-mib',type=int,default=45000)
    args = p.parse_args(); m = json.loads(args.manifest.read_text())
    assert args.max_new_groups>=0
    assert m['protocol']=='c4_unseen_rules_v1' and m['conditions']==list(CONDITIONS)
    assert sha(args.adapter/'adapter_model.safetensors')==m['checkpoints'][args.arm]['adapter_sha256']
    expected = {
        'config.json':'97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb',
        'model.safetensors.index.json':'bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0',
        'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    }
    for name, checksum in expected.items(): assert sha(args.model/name)==checksum
    source = Path(__file__).parent
    code = {name:sha(source/name) for name in (
        'evaluate_c4_unseen_rules.py','c4_unseen_rules.py','c4_execution_model.py',
        'c4_execution_data.py','c4_rule_followup.py','c4_freegen_protocol.py',
        'train_c1_lora.py','state_swap_misconception_smoke.py')}
    fp = dict(arm=args.arm,manifest_sha256=sha(args.manifest),
              adapter_sha256=m['checkpoints'][args.arm]['adapter_sha256'],
              base=expected,code=code,precision='same NF4/bfloat16 kbit preparation as training eval')
    out = args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'lineage.json').exists(): assert json.loads((out/'lineage.json').read_text())==fp
    atomic(out/'lineage.json',fp)
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(args.model,local_files_only=True); tok.pad_token=tok.eos_token
    # Full preflight checks every prompt length; never silently truncate.
    for g in m['groups']:
        for q in g['queries']:
            for c in CONDITIONS:
                text = prompt(g,q,c)
                assert digest(text)==g['prompt_hashes'][f'{q["query_idx"]}:{c}']
                ids,_,_=locate_bottleneck_positions(tok,text,'<|vision_pad|>')
                assert len(ids)+m['max_new_tokens']<=m['max_length']
    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    assert free>=args.min_free_mib,('Insufficient local GPU headroom',free,args.min_free_mib)
    load_start=time.monotonic()
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        quantization_config=quant,torch_dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='eager')
    # Preserve float casting from the prior endpoint; no optimizer is created.
    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model,str(args.adapter),is_trainable=False)
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
    assert all(config.r==8 for config in model.peft_config.values())
    assert len(get_decoder_layers(model))==64
    model.eval(); all_records=[]; timings=[]
    runtime=dict(arm=args.arm,initial_free_mib=free,load_seconds=time.monotonic()-load_start,
        versions={name:importlib.metadata.version(name) for name in
                  ('torch','transformers','peft','bitsandbytes','accelerate','safetensors')},
        gpu=torch.cuda.get_device_name(0),fingerprint=fp)
    for gi,g in enumerate(m['groups']):
        path = out/(g['id'].replace(':','_')+'.json')
        if path.exists():
            previous = json.loads(path.read_text()); assert previous['fingerprint']==fp
            want={(q['query_idx'],c) for q in g['queries'] for c in CONDITIONS}
            assert len(previous['records'])==len(want)
            assert {(r['query_idx'],r['condition']) for r in previous['records']}==want
            all_records.extend(previous['records']); continue
        started=time.monotonic()
        states={role:capture(model,tok,INSTRUCTION+h+STATE) for role,h in g['histories'].items()}
        import hashlib
        hashes={role:hashlib.sha256(b''.join(v.float().cpu().numpy().tobytes() for v in vs)).hexdigest()
                for role,vs in states.items()}
        rs=[]
        for q in g['queries']:
            for c in CONDITIONS:
                text=prompt(g,q,c)
                role={'noop':'recipient','swap':'donor','random':'random','same_donor':'same_donor'}.get(c)
                y=(open_generate(model,tok,text,m['max_new_tokens']) if c=='open_history'
                   else generate(model,tok,text,states[role] if role else None,m['max_new_tokens']))
                if y['stop']=='token_limit': y['parsed']=None; y['parse_type']='truncated'
                rs.append(dict(group_id=g['id'],pair_id=g['pair_id'],category=g['category'],
                    query_idx=q['query_idx'],condition=c,prompt_sha256=digest(text),
                    targets=q['targets'],correct=q['correct'],**y))
        atomic(path,dict(fingerprint=fp,state_hashes=hashes,records=rs))
        all_records.extend(rs)
        elapsed=time.monotonic()-started
        timings.append(dict(group_id=g['id'],seconds=elapsed,records=len(rs)))
        if args.runtime_report:
            atomic(args.runtime_report,dict(runtime,groups=timings,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        print(f'[unseen-rules] {args.arm} {gi+1}/{len(m["groups"])} seconds={elapsed:.1f}',flush=True)
        if args.max_new_groups and len(timings)>=args.max_new_groups:
            if len(all_records)<m['n_generations_per_checkpoint']:
                print('[unseen-rules] bounded preflight complete; full evaluation is not complete',flush=True)
                return
    assert len(all_records)==m['n_generations_per_checkpoint']
    atomic(out/'records.json',all_records)
    atomic(out/'complete.json',dict(status='complete',fingerprint=fp,
        records=len(all_records),records_sha256=sha(out/'records.json')))


if __name__=='__main__': main()
