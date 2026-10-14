"""CPU data and interface checks; synthetic tests are not model evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from c4_unseen_rules import (RULES, PAIRS, CONDITIONS, answer, reference, correct,
                            old_sources, prompt, digest)
from c4_freegen_protocol import signatures, numeric_tokens


@pytest.fixture(scope='module')
def manifest():
    return json.loads((ROOT/'manifests/c4_unseen_rules_v1_20260906.json').read_text())


def test_reference_formula_examples():
    q=dict(operands=[2,7,3,7])
    want={'offset_3':'9','offset_7':'13','shift_1_keep':'12/7','shift_2_keep':'20/7',
          'even_ignore_odd_keep':'6/7','even_keep_odd_ignore':'6',
          'seen_ignore':'6','seen_keep':'6/7'}
    for rule, value in want.items():
        assert answer(rule,q)==reference(rule,q)==value
    q['operands']=[2,8,3,8]
    assert answer('even_ignore_odd_keep',q)=='6'
    assert answer('even_keep_odd_ignore',q)=='3/4'


def test_panel_isolation_counts_and_rule_identifiability(manifest):
    m=manifest; seen,_=old_sources(ROOT); new=set(); pair_ids=set()
    assert m['n_queries']==96 and m['n_generations_per_checkpoint']==864
    assert len(m['groups'])==32
    for g in m['groups']:
        assert len(set(g['histories'].values()))==4
        assert g['rules']['random']==g['rules']['recipient']
        assert g['rules']['same_donor']==g['rules']['donor']
        for role,events in g['history_events'].items():
            compatible=[r for r in RULES if all(answer(r,q)==answer(g['rules'][role],q) for q in events)]
            assert compatible==[g['rules'][role]]
            assert {q['operands'][1]%2 for q in events}=={0,1}
        assert {q['operands'][1]%2 for q in g['queries']}=={0,1}
        sig_rules=[r for r in RULES if all(answer(r,q)==q['targets']['donor'] for q in g['queries'])]
        assert sig_rules==[g['rules']['donor']]
        visible=set(numeric_tokens(' '.join(g['histories'].values())))
        for q in g['queries']:
            assert q['targets']['donor']!=q['targets']['recipient']
            assert q['correct']==correct(q)
            for r,y in q['targets'].items():
                assert y==answer(g['rules'][r],q)==reference(g['rules'][r],q)
                assert y not in visible and y!=q['correct']
        for role in ('donor','recipient'):
            assert len({q['targets'][role] for q in g['queries']})==3
        # Two directions intentionally share the same pair's questions/histories.
        if g['pair_id'] in pair_ids: continue
        pair_ids.add(g['pair_id'])
        for q in [q for events in g['history_events'].values() for q in events]+g['queries']:
            sig=signatures(q)
            assert not sig&seen and not sig&new
            new.update(sig)
    assert len(pair_ids)==16
    for category in PAIRS:
        assert len({g['pair_id'] for g in m['groups'] if g['category']==category})==4


def test_prompt_separation_and_lengths(manifest):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT.parents[2]/'models/Qwen3-32B',local_files_only=True)
    maximum=0
    for g in manifest['groups']:
        for q in g['queries']:
            for c in CONDITIONS:
                text=prompt(g,q,c)
                assert digest(text)==g['prompt_hashes'][f'{q["query_idx"]}:{c}']
                query=text.split('<query>')[1]
                assert 'Valid outputs:' not in query and 'Correct answer:' not in query
                assert ('Student rule:' in query)==(c=='oracle')
                if c in ('oracle','empty_history'):
                    assert 'Trial 1:' not in text
                # Targets are evaluator-only metadata, not renderer inputs.
                changed=copy.deepcopy(q); changed['targets']={'donor':'SENTINEL'}
                changed['correct']='SENTINEL'
                assert prompt(g,changed,c)==text
                n=len(tok(text.replace('<STUDENT_STATE>','<|vision_pad|>'),add_special_tokens=False)['input_ids'])
                maximum=max(maximum,n)
                assert n+manifest['max_new_tokens']<=manifest['max_length']
    assert maximum>0


def test_collector_synthetic_controls_and_corruption(manifest):
    from collect_c4_unseen_rules import audit_records, summarize
    class SyntheticTokenizer:
        def decode(self,tokens,skip_special_tokens=False):return ''.join(map(chr,tokens))
    records=[]
    for g in manifest['groups']:
        for q in g['queries']:
            for c in CONDITIONS:
                y=q['targets']['donor'] if c=='swap' else q['targets']['recipient']
                text=f'<predicted_response>{y}</predicted_response>'
                records.append(dict(group_id=g['id'],pair_id=g['pair_id'],category=g['category'],
                    query_idx=q['query_idx'],condition=c,targets=q['targets'],correct=q['correct'],
                    prompt_sha256=g['prompt_hashes'][f'{q["query_idx"]}:{c}'],
                    text=text,tokens=list(map(ord,text)),parsed=y,parse_type='tagged',stop='closing_tag'))
    # Synthetic character tokenizer needs more than the actual BPE token budget.
    m=copy.deepcopy(manifest); m['max_new_tokens']=128
    tok=SyntheticTokenizer(); idx=audit_records(m,records,tok)
    out=summarize(m,dict(A=idx,B=idx),draws=100)
    for category in PAIRS:
        assert out['arms']['A'][category]['specificity']['estimate']==1
        assert out['B_minus_A_by_category'][category]['estimate']==0
    with pytest.raises(AssertionError):audit_records(m,records+records[:1],tok)
    with pytest.raises(AssertionError):audit_records(m,records[:-1],tok)
    broken=copy.deepcopy(records); broken[0]['parsed']='SENTINEL'
    with pytest.raises(AssertionError):audit_records(m,broken,tok)


def test_open_history_attention_is_the_only_decoder_difference(monkeypatch):
    import torch
    from types import SimpleNamespace
    import evaluate_c4_unseen_rules as evaluator
    import c4_execution_model as bottleneck
    class Tok:
        eos_token_id=0
        def decode(self,tokens,skip_special_tokens=False):return '5' if tokens else ''
    class Model:
        generation_config=SimpleNamespace(eos_token_id=0)
        def __init__(self):self.masks=[]
        def get_input_embeddings(self):return SimpleNamespace(weight=torch.zeros(1))
        def __call__(self,input_ids,attention_mask,**kwargs):
            self.masks.append(attention_mask.clone())
            logits=torch.zeros((1,input_ids.shape[1],3));logits[...,1 if len(self.masks)==1 else 0]=10
            return SimpleNamespace(logits=logits,past_key_values='synthetic')
    positions=lambda *args:([1]*9,1,4)
    monkeypatch.setattr(evaluator,'locate_bottleneck_positions',positions)
    monkeypatch.setattr(bottleneck,'locate_bottleneck_positions',positions)
    om,bm=Model(),Model()
    a=evaluator.open_generate(om,Tok(),'unused',48)
    b=bottleneck.generate(bm,Tok(),'unused',None,48)
    assert a==b and a['parsed']=='5'
    assert om.masks[0][0,0,8,2]==0
    assert bm.masks[0][0,0,8,2]<-1e20
    assert bm.masks[0][0,0,8,4]==0 # Answer anchor sees state.
    assert bm.masks[0][0,0,6,4]<-1e20 # Query does not.
    assert om.masks[1][0,0,0,2]==0 and bm.masks[1][0,0,0,2]<-1e20
