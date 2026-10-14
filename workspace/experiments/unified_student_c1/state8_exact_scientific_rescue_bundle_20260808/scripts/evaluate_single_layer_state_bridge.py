#!/usr/bin/env python3
"""Frozen C4 bridge using one hidden-size vector in one native layer space."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from evaluate_layerwise_state_bridge import activate_decode_adapter, closed_position_ids
from recurrent_student_model import build_closed_decode_attention_mask, extract_direct_decode_text
from state_swap_misconception_smoke import (
    build_bottleneck_attention_mask,
    encode_prompt_text,
    family_of,
    get_decoder_layers,
    label_of,
    load_model,
    load_pair_manifest,
    locate_bottleneck_positions,
    locate_state_position,
    pair_targets,
    read_jsonl,
)


CONDITIONS = ("noop", "samemal", "random", "swap")


@torch.inference_mode()
def capture_state(model, tokenizer, row, alias: str, layer: int, max_length: int):
    ids, history_start, state_position = locate_bottleneck_positions(
        tokenizer, str(row["prompt"]), alias
    )
    if len(ids) > max_length:
        raise ValueError("Prompt is too long")
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_2d = torch.ones_like(input_ids)
    attention = build_bottleneck_attention_mask(
        attention_2d,
        torch.tensor([history_start], device=device),
        torch.tensor([state_position], device=device),
        torch.tensor([len(ids)], device=device),
        "late_fusion",
    )
    captured = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0, state_position].detach().float().cpu())

    handle = get_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Single-layer state capture failed")
    return captured[0], state_position


def random_state(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    noise = torch.randn(reference.shape, generator=generator)
    return noise * (
        torch.linalg.vector_norm(reference.float())
        / torch.linalg.vector_norm(noise).clamp_min(1e-12)
    )


@torch.inference_mode()
def score_state(
    model, tokenizer, state, source_position: int, row, alias: str,
    layer: int, max_length: int, position_mode: str,
):
    text = extract_direct_decode_text(str(row["prompt"]))
    if "<student_history>" in text.lower():
        raise AssertionError("Single-layer decoder leaked raw history")
    aliased = encode_prompt_text(text, alias)
    prefix_ids, state_position = locate_state_position(tokenizer, text, alias)
    if prefix_ids != tokenizer(aliased, add_special_tokens=False)["input_ids"]:
        raise AssertionError("Closed prompt tokenization drifted")
    items=[]
    for label_value in row["candidate_labels"]:
        label=str(label_value)
        target_ids=tokenizer(str(row["candidate_targets"][label]),add_special_tokens=False)["input_ids"]
        items.append((label,prefix_ids+target_ids,len(prefix_ids)))
    device=next(model.parameters()).device; pad=tokenizer.pad_token_id or tokenizer.eos_token_id
    maximum=max(len(ids) for _,ids,_ in items)
    input_ids=torch.tensor([ids+[pad]*(maximum-len(ids)) for _,ids,_ in items],device=device)
    attention_2d=torch.tensor([[1]*len(ids)+[0]*(maximum-len(ids)) for _,ids,_ in items],device=device)
    prompt_ends=torch.tensor([end for _,_,end in items],device=device)
    carrier=torch.full((len(items),),state_position,device=device,dtype=torch.long)
    attention=build_closed_decode_attention_mask(attention_2d,carrier,prompt_ends)
    positions=closed_position_ids(maximum,state_position,source_position,position_mode,device).expand(len(items),-1)

    def hook(_module,_inputs,output):
        hidden=output[0] if isinstance(output,tuple) else output
        patched=hidden.clone(); patched[:,state_position,:]=state.to(hidden.device,dtype=hidden.dtype)
        return (patched,)+output[1:] if isinstance(output,tuple) else patched
    handle=get_decoder_layers(model)[layer].register_forward_hook(hook)
    try: logits=model(input_ids=input_ids,attention_mask=attention,position_ids=positions,use_cache=False).logits
    finally: handle.remove()
    scores={}
    for i,(label,ids,end) in enumerate(items):
        score=0.0
        for pos in range(end,len(ids)):
            score+=float(torch.log_softmax(logits[i,pos-1].float(),dim=-1)[input_ids[i,pos]].item())
        scores[label]=score
    labels=list(scores); probs=torch.softmax(torch.tensor([scores[x] for x in labels]),dim=0).tolist()
    return {"prediction":max(scores,key=scores.get),"scores":scores,"probabilities":dict(zip(labels,probs,strict=True))}


def summarize(records):
    by=defaultdict(dict)
    for r in records:by[int(r['pair_idx'])][r['condition']]=r
    required={'native',*CONDITIONS}
    bad=[i for i,x in by.items() if set(x)!=required]
    if bad:raise ValueError(f"Incomplete conditions: {bad[:5]}")
    clean=sorted(i for i,x in by.items() if x['native']['category']=='stay_B')
    if not clean:raise ValueError('No clean pairs')
    native=[by[i]['native'] for i in clean]
    out={'clean_n':len(clean),'all_pairs_n':len(by),'clean':{}}
    for cond in CONDITIONS:
        rows=[by[i][cond] for i in clean]; counts=Counter(r['category'] for r in rows);n=len(rows)
        out['clean'][cond]={'n':n,'category_counts':dict(counts),'flip_to_A_rate':counts['flip_to_A']/n,'stay_B_rate':counts['stay_B']/n,'other_rate':counts['other']/n,'changed_top1_rate':sum(r['prediction']!=b['prediction'] for r,b in zip(rows,native,strict=True))/n}
    out['D_specific']=out['clean']['swap']['flip_to_A_rate']-out['clean']['random']['flip_to_A_rate']
    out['D_damage']=out['clean']['random']['flip_to_A_rate']-out['clean']['noop']['flip_to_A_rate']
    checks={'candidate_support':len(clean)>=50,'specificity':out['D_specific']>=.15,'noop_control':out['clean']['noop']['flip_to_A_rate']<=.05,'same_rule_control':out['clean']['samemal']['flip_to_A_rate']<=.10}
    out['checks']=checks;out['scientific_gate_pass']=all(checks.values());return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--decode-adapter');p.add_argument('--jsonl',type=Path,required=True);p.add_argument('--pair-manifest',type=Path,required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--layer',type=int,required=True);p.add_argument('--position-mode',choices=('compact','one_shot_aligned'),default='compact');p.add_argument('--state-token-alias',default='<|vision_pad|>');p.add_argument('--max-length',type=int,default=1536);p.add_argument('--num-pairs',type=int,default=150);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    rows=read_jsonl(args.jsonl,'malrule_next_response');pairs,controls=load_pair_manifest(args.pair_manifest,rows,'malrule_next_response','response',args.num_pairs)
    tokenizer,model=load_model(args.model,args.adapter,True,args.state_token_alias); n_layers=len(get_decoder_layers(model))
    if not 0<=args.layer<n_layers:raise ValueError('Layer out of range')
    unique={}
    for a,b in pairs:unique[str(a['example_id'])]=a;unique[str(b['example_id'])]=b
    for c in controls:
        if c is not None:unique[str(c['example_id'])]=c
    states={};positions={}
    for i,(key,row) in enumerate(unique.items(),1):
        states[key],positions[key]=capture_state(model,tokenizer,row,args.state_token_alias,args.layer,args.max_length)
        if i==1 or i%10==0 or i==len(unique):print(json.dumps({'states':i,'total':len(unique)}),flush=True)
    decode_adapter=args.decode_adapter or args.adapter;active=activate_decode_adapter(model,args.adapter,decode_adapter);records=[]
    for idx,((donor,recipient),control) in enumerate(zip(pairs,controls,strict=True),1):
        a,b,_=pair_targets(donor,recipient,'response');rid=str(recipient['example_id']); conditions={'native':states[rid],'noop':states[rid].clone(),'swap':states[str(donor['example_id'])],'random':random_state(states[rid],args.seed+1000003*idx),'samemal':states[str(control['example_id'])]}
        for cond,state in conditions.items():
            scored=score_state(model,tokenizer,state,positions[rid],recipient,args.state_token_alias,args.layer,args.max_length,args.position_mode);pred=scored['prediction'];category='flip_to_A' if pred==a else ('stay_B' if pred==b else 'other')
            records.append({'pair_idx':idx,'condition':cond,'category':category,'prediction':pred,'donor_id':donor['example_id'],'recipient_id':recipient['example_id'],'donor_label':label_of(donor),'recipient_label':label_of(recipient),'family':family_of(recipient),'donor_target':a,'recipient_target':b})
        if idx==1 or idx%10==0:print(json.dumps({'pairs':idx,'total':len(pairs)}),flush=True)
    result={'model':args.model,'capture_adapter':args.adapter,'decode_adapter':decode_adapter,'active_decode_adapter':active,'layer':args.layer,'position_mode':args.position_mode,'state_shape':[int(states[next(iter(states))].numel())],'intervention_target':'single_native_layer_state_vector','raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False,'optimizer_steps':0,'summary':summarize(records),'records':records}
    args.output_json.parent.mkdir(parents=True,exist_ok=True);args.output_json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result['summary'],indent=2,sort_keys=True))


if __name__=='__main__':main()
