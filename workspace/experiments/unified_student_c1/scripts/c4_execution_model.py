"""Differentiable same-forward state exchange; no detached donor cache in training."""
from contextlib import contextmanager
import torch
from train_c1_lora import build_bottleneck_attention_mask, encode_example
from state_swap_misconception_smoke import get_decoder_layers, locate_bottleneck_positions

ALIAS='<|vision_pad|>'

def encode(tokenizer,row,arm,max_length=768):
    target_prompt=row['recipient'] if arm=='B' and row['mode']=='history' else row['native']
    raw_length=len(tokenizer(target_prompt.replace('<STUDENT_STATE>',ALIAS),add_special_tokens=False)['input_ids'])+len(tokenizer(row['target'],add_special_tokens=False)['input_ids'])+1
    assert raw_length<=max_length,('No truncation permitted',raw_length,max_length)
    t=encode_example(tokenizer,dict(prompt=target_prompt,target=row['target']),max_length,True,True,ALIAS)
    ids,h0,sp=locate_bottleneck_positions(tokenizer,row['donor_prefix'],ALIAS)
    assert sp==len(ids)-1
    source=dict(input_ids=ids,attention_mask=[1]*len(ids),labels=[-100]*len(ids),history_start=h0,state_pos=sp,prompt_end=len(ids))
    n=max(len(ids),len(t['input_ids']))
    assert n<=max_length
    packed={k:torch.tensor([v[k]+[pad]*(n-len(v[k])) for v in (source,t)])
            for k,pad in (('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100))}
    # Source contains only the history prefix; its state reads history normally.
    mask=build_bottleneck_attention_mask(packed['attention_mask'],torch.tensor([h0,t['history_start']]),
             torch.tensor([sp,t['state_pos']]),style='standard')
    end=t['prompt_end'];pos=t['state_pos']
    mask[1,0,pos+1:end-1,pos]=torch.finfo(torch.float32).min
    packed['attention_mask']=mask
    return packed,(sp,pos)

@contextmanager
def exchange(model,positions,enabled=True):
    """Hooks are pure and stay installed through backward/checkpoint recompute.

    Source row 0 is unchanged; row 1 receives row 0's state at each block.
    Clone/assignment preserves autograd. No donor future query is encoded.
    """
    handles=[]
    if enabled:
        src,dst=positions
        def hook(module,inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            changed=h.clone();changed[1,dst,:]=h[0,src,:]
            return (changed,)+output[1:] if isinstance(output,tuple) else changed
        handles=[layer.register_forward_hook(hook) for layer in get_decoder_layers(model)]
    try: yield
    finally:
        for h in handles:h.remove()

@contextmanager
def patch(model,position,vectors):
    handles=[]
    if vectors is not None:
        for layer,vec in zip(get_decoder_layers(model),vectors):
            def hook(module,inputs,output,v=vec):
                x=output[0] if isinstance(output,tuple) else output
                h=x.clone();h[0,position,:]=v.to(h)
                return (h,)+output[1:] if isinstance(output,tuple) else h
            handles.append(layer.register_forward_hook(hook))
    try:yield
    finally:
        for h in handles:h.remove()

@torch.no_grad()
def capture(model,tokenizer,prefix):
    ids,h0,sp=locate_bottleneck_positions(tokenizer,prefix,ALIAS)
    t=torch.tensor([ids],device=model.get_input_embeddings().weight.device)
    mask=build_bottleneck_attention_mask(torch.ones_like(t),torch.tensor([h0]),torch.tensor([sp]),style='standard')
    vectors=[]
    def hook(module,inputs,output):
        h=output[0] if isinstance(output,tuple) else output
        vectors.append(h[0,sp].detach().clone())
    hs=[l.register_forward_hook(hook) for l in get_decoder_layers(model)]
    try:model(input_ids=t,attention_mask=mask,use_cache=False)
    finally:
        for h in hs:h.remove()
    return vectors

@torch.no_grad()
def generate(model,tokenizer,prompt,vectors=None,limit=48):
    from c4_freegen_protocol import parse_output
    ids,h0,sp=locate_bottleneck_positions(tokenizer,prompt,ALIAS)
    device=model.get_input_embeddings().weight.device
    tokens=[];past=None;stop='token_limit'
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id}
    for step in range(limit):
        if step==0:
            t=torch.tensor([ids],device=device)
            mask=build_bottleneck_attention_mask(torch.ones_like(t),torch.tensor([h0]),torch.tensor([sp]),torch.tensor([len(ids)]),'late_fusion')
        else:
            t=torch.tensor([[tokens[-1]]],device=device)
            mask=torch.zeros((1,1,1,len(ids)+step),device=device)
            mask[...,h0:sp]=torch.finfo(torch.float32).min
        with patch(model,sp,vectors if step==0 else None):
            out=model(input_ids=t,attention_mask=mask,past_key_values=past,use_cache=True)
        past=out.past_key_values;token=int(out.logits[0,-1].argmax());del out
        if token in eos:stop='eos';break
        tokens.append(token)
        if '</predicted_response>' in tokenizer.decode(tokens,skip_special_tokens=False):stop='closing_tag';break
    text=tokenizer.decode(tokens,skip_special_tokens=False)
    parsed,kind=parse_output(text)
    return dict(tokens=tokens,text=text,parsed=parsed,parse_type=kind,stop=stop,prompt_tokens=len(ids))
