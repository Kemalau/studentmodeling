import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from c4_history_diagnostic import COHORTS, CONDITIONS, MANIFEST, SOURCE, build, validate
from c4_unseen_rules import answer, prompt
from c4_freegen_protocol import numeric_tokens, signatures
from collect_c4_history_diagnostic import audit_records, summarize, render


@pytest.fixture(scope='module')
def panel():
    return json.loads((ROOT/MANIFEST).read_text())


def test_frozen_manifest_and_nested_controls(panel):
    old=json.loads((ROOT/SOURCE).read_text())
    validate(panel,old)
    assert panel==build()
    assert len(panel['groups'])*3*len(CONDITIONS)*2==5376
    index={(g['cohort'],g['original_group_id']):g for g in panel['groups']}
    for g in panel['groups']:
        h3=index['h3',g['original_group_id']]
        h12=index['h12',g['original_group_id']]
        assert g['queries']==h3['queries']
        for q in g['queries']:
            assert prompt(g,q,'oracle')==prompt(h3,q,'oracle')
        for r in g['history_events']:
            assert h12['history_events'][r][:3]==h3['history_events'][r]
        reverse_id=g['original_group_id'][:-1]+str(1-int(g['original_group_id'][-1]))
        opposite=index[g['cohort'],reverse_id]
        assert g['histories']['donor']==opposite['histories']['recipient']


def test_added_history_has_no_query_or_target_leakage(panel):
    from c4_unseen_rules import old_sources
    old=json.loads((ROOT/SOURCE).read_text())
    excluded,_=old_sources(ROOT)
    for g in old['groups']:
        for es in g['history_events'].values():
            for q in es:excluded.update(signatures(q))
        for q in g['queries']:excluded.update(signatures(q))
    added=set()
    for g in panel['groups']:
        if g['cohort']!='h12' or not g['id'].endswith(':0'):continue
        forbidden={v for q in g['queries'] for v in q['targets'].values()}
        assert not forbidden & set(numeric_tokens(' '.join(g['histories'].values())))
        for es in g['history_events'].values():
            for q in es[3:]:
                sig=signatures(q)
                assert not sig & (excluded|added)
                added.update(sig)


def test_all_prompt_lengths_and_checkpoint_hashes(panel):
    from transformers import AutoTokenizer
    from c4_freegen_protocol import sha
    tok=AutoTokenizer.from_pretrained('/home/v-yuejiali/studentmodeling/models/Qwen3-32B',local_files_only=True)
    for g in panel['groups']:
        for q in g['queries']:
            for c in CONDITIONS:
                text=prompt(g,q,c).replace('<STUDENT_STATE>','<|vision_pad|>')
                assert len(tok(text,add_special_tokens=False)['input_ids'])+48<=768
    root=Path('/datadisk/v-yuejiali/studentmodeling/c4_unseen_rules_20260906/checkpoints/c4-execution-training-v1-seed42-recovery1-20260906')
    for arm in ('A','B'):
        file=root/f'c4-execution-arm-{arm.lower()}-seed42'/f'execution_{arm}'/'final_adapter'/'adapter_model.safetensors'
        assert sha(file)==panel['checkpoints'][arm]['adapter_sha256']


def synthetic(panel):
    from transformers import AutoTokenizer
    from c4_freegen_protocol import parse_output
    tok=AutoTokenizer.from_pretrained('/home/v-yuejiali/studentmodeling/models/Qwen3-32B',local_files_only=True)
    records=[]
    for g in panel['groups']:
        for q in g['queries']:
            for c in CONDITIONS:
                success=c=='oracle' or (g['cohort']=='h12' and c in ('swap','donor_native','open_history'))
                text='<predicted_response>'+q['targets']['donor' if success else 'recipient']+'</predicted_response>'
                parsed,kind=parse_output(text)
                length=len(tok(prompt(g,q,c).replace('<STUDENT_STATE>','<|vision_pad|>'),add_special_tokens=False)['input_ids'])
                records.append(dict(group_id=g['id'],original_group_id=g['original_group_id'],cohort=g['cohort'],
                    pair_id=g['pair_id'],category=g['category'],query_idx=q['query_idx'],condition=c,
                    targets=q['targets'],correct=q['correct'],prompt_sha256=g['prompt_hashes'][f'{q["query_idx"]}:{c}'],
                    prompt_tokens=length,text=text,tokens=tok(text,add_special_tokens=False)['input_ids'],
                    parsed=parsed,parse_type=kind,stop='closing_tag'))
    return tok,records


def test_collector_rejects_missing_duplicate_and_corrupt_records(panel):
    tok,records=synthetic(panel)
    rs=[r for r in records if r['cohort']=='h3']
    assert len(audit_records(panel,rs,tok,'h3'))==672
    for bad in (rs[:-1],rs+[rs[0]]):
        with pytest.raises(AssertionError):audit_records(panel,bad,tok,'h3')
    for field,value in (('prompt_sha256','bad'),('parsed','999'),('prompt_tokens',0),('pair_id','fake')):
        bad=copy.deepcopy(rs);bad[0][field]=value
        with pytest.raises(AssertionError):audit_records(panel,bad,tok,'h3')


def test_paired_contrasts_group_denominators_and_report(panel):
    tok,records=synthetic(panel);idx={}
    for cohort in COHORTS:
        idx.update(audit_records(panel,[r for r in records if r['cohort']==cohort],tok,cohort))
    result=summarize(panel,{'A':idx,'B':idx},draws=40)
    for arm in ('A','B'):
        for category in result['arms'][arm]:
            r=result['arms'][arm][category]['h12']
            assert r['specificity']['estimate']==1
            assert r['specificity']['n_independent_pairs']==4
            assert r['counts']['swap']['n']==24
            assert r['joint3']['swap']['all3_donor_hits']==8
            for comparison in ('h12_minus_h3','h12_minus_repeat3x4'):
                assert result['paired_history_contrasts'][arm][category][comparison]['specificity']['estimate']==1
    assert 'No-op mismatches' in render(result)


def test_submission_plan_and_fail_closed_empty_audit():
    from prepare_c4_history_diagnostic_feeds import build, FILES
    from submit_c4_history_diagnostic import resolve_empty
    config=build()
    assert len(config['jobs'])==8
    assert len({j['name'] for j in config['jobs']})==8
    assert all(j['sku']=='80G1-A100' for j in config['jobs'])
    assert all((ROOT/f).is_file() for f in FILES)
    bad=dict(errors=[dict(experiment='x',reason='unknown state')])
    with pytest.raises(AssertionError):resolve_empty(bad,[])
    empty=dict(errors=[dict(experiment='x',reason='missing JSON table: x')])
    with pytest.raises(AssertionError):resolve_empty(copy.deepcopy(empty),[dict(experiment_name='x',n_jobs=1,_all_terminated=True)])
    ok=resolve_empty(copy.deepcopy(empty),[dict(experiment_name='x',n_jobs=0,_all_terminated=True)])
    assert ok['auditable'] and len(ok['resolved_empty_experiments'])==1
