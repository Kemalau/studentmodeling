"""CPU data/gradient/causal-path tests, not trained-model evidence."""
import copy
import json
from pathlib import Path
import sys
import os
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from c4_execution_data import rows,prompt,answer,correct
from c4_execution_model import exchange,encode,generate,ALIAS
from c4_freegen_protocol import signatures,numeric_tokens
from train_c1_lora import build_bottleneck_attention_mask

def tokenizer_path():
    """Evaluate the local fallback only when no remote path was supplied."""
    supplied=os.environ.get('C4_TEST_TOKENIZER')
    if supplied:
        return supplied
    if len(ROOT.parents)<3:
        raise RuntimeError('Set C4_TEST_TOKENIZER to the staged model directory')
    return str(ROOT.parents[2]/'models/Qwen3-32B')

def test_remote_tokenizer_override_does_not_evaluate_local_fallback():
    from unittest.mock import patch
    with patch.dict(os.environ,{'C4_TEST_TOKENIZER':'/tmp/Qwen3-32B'}), patch(__name__+'.ROOT',Path('/scratch/amlt_code')):
        assert tokenizer_path()=='/tmp/Qwen3-32B'
    with patch.dict(os.environ,{},clear=True), patch(__name__+'.ROOT',Path('/scratch/amlt_code')):
        try:tokenizer_path()
        except RuntimeError as exc:assert 'C4_TEST_TOKENIZER' in str(exc)
        else:raise AssertionError('Missing remote tokenizer path must fail clearly')

def test_manifest_isolation_and_matched_targets():
    m=json.loads((ROOT/'manifests/c4_execution_training_v1_20260906.json').read_text())
    used=set();histories=set()
    for split,gs in m['splits'].items():
        numbers=set();hs=set()
        for g in gs:
            assert len(set(g['histories'].values()))==4
            hs.update(g['histories'].values())
            for events in g['history_events'].values():
                for q in events:numbers.update(signatures(q))
            for q in g['queries']:
                numbers.update(signatures(q))
                assert q['targets']['donor']!=q['targets']['recipient']
                for role,rule in g['rules'].items(): assert q['targets'][role]==answer(rule,q)
                assert q['correct']==correct(q)
                for y in q['targets'].values():assert y not in numeric_tokens(' '.join(g['histories'].values()))
                s=prompt(g['histories']['recipient'],q).split('<query>')[1]
                assert 'Valid outputs:' not in s and 'Correct answer:' not in s
        assert not (used&numbers) and not (histories&hs)
        used.update(numbers);histories.update(hs)
    rs=rows(m,'train');assert len(rs)==1920
    assert all(r['mode']=='oracle' for r in rs[:960])
    assert all(r['mode']=='history' for r in rs[960:])
    tok=AutoTokenizer.from_pretrained(tokenizer_path(),local_files_only=True)
    for r in rs[::19]:
        a,pa=encode(tok,r,'A');b,pb=encode(tok,r,'B')
        assert torch.equal(a['labels'][a['labels']!=-100],b['labels'][b['labels']!=-100])
        assert (b['labels'][0]==-100).all()
        # Train and decode both open answer anchor to state.
        anchor=int((b['labels'][1]!=-100).nonzero()[0])-1
        assert b['attention_mask'][1,0,anchor,pb[1]]==0

def tiny():
    c=Qwen3Config(vocab_size=41,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,attention_dropout=0.)
    c._attn_implementation='eager';return Qwen3ForCausalLM(c)

def test_exchange_gradients_survive_checkpointing():
    torch.manual_seed(7);base=tiny();other=copy.deepcopy(base)
    ids=torch.tensor([[2,3,4,5,6,0,0,0,0],[2,10,11,12,6,13,14,15,16]])
    att=ids.ne(0).long();mask=build_bottleneck_attention_mask(att,torch.tensor([1,1]),torch.tensor([4,4]),style='standard')
    mask[1,0,5:6,4]=torch.finfo(torch.float32).min
    labels=torch.full_like(ids,-100);labels[1,7:]=ids[1,7:]
    grads=[]
    for model,checkpoint in ((base,False),(other,True)):
        model.train()
        if checkpoint:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        with exchange(model,(4,4)):
            loss=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False).loss
            loss.backward()
        g=model.model.embed_tokens.weight.grad
        assert float(g[3].abs().sum())>0,'Source history must receive gradient through exchange'
        assert float(g[10].abs().sum())<1e-10,'Recipient hidden history must not leak'
        grads.append(g)
    torch.testing.assert_close(grads[0],grads[1],rtol=2e-4,atol=2e-6)
    # A control does not backpropagate into unused source-only tokens.
    base.zero_grad()
    base(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False).loss.backward()
    assert float(base.model.embed_tokens.weight.grad[3].abs().sum())==0

def test_same_forward_swap_equals_separate_capture_and_patch():
    from c4_execution_model import patch
    torch.manual_seed(14);model=tiny().eval()
    ids=torch.tensor([[2,3,4,5,6,0,0,0,0],[2,10,11,12,6,13,14,15,16]])
    mask=build_bottleneck_attention_mask(ids.ne(0).long(),torch.tensor([1,1]),torch.tensor([4,4]),style='standard')
    with torch.no_grad(),exchange(model,(4,4)):
        together=model(input_ids=ids,attention_mask=mask).logits[1]
    vs=[]
    hs=[l.register_forward_hook(lambda m,i,o:vs.append(o[0,4].detach().clone())) for l in model.model.layers]
    with torch.no_grad():model(input_ids=ids[:1,:5])
    for h in hs:h.remove()
    with torch.no_grad(),patch(model,4,vs):
        separate=model(input_ids=ids[1:],attention_mask=mask[1:]).logits[0]
    torch.testing.assert_close(together[5:],separate[5:],rtol=1e-4,atol=1e-6)

if __name__=='__main__':
    test_remote_tokenizer_override_does_not_evaluate_local_fallback()
    test_manifest_isolation_and_matched_targets()
    test_exchange_gradients_survive_checkpointing()
    test_same_forward_swap_equals_separate_capture_and_patch()
    print('Four protocol/gradient/path tests passed',flush=True)
