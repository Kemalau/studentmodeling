"""Synthetic audit/report checks; not model performance evidence."""
import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_c4_unseen_rules_delivery import frozen_code_checks,render
from c4_unseen_rules import CONDITIONS,PAIRS


def test_finalizer_wait_requires_live_collector():
    from finalize_c4_unseen_rules import ready,EXP
    receipt=dict(status='watching',experiment=EXP,pid=123)
    assert not ready(receipt,lambda pid:True)
    with pytest.raises(RuntimeError):ready(receipt,lambda pid:False)
    with pytest.raises(RuntimeError):ready(dict(receipt,status='needs_attention'))
    assert ready(dict(receipt,status='complete'))


def test_frozen_code_must_match_exact_snapshot():
    frozen={line.split('  ',1)[1]:line.split('  ',1)[0] for line in
        (ROOT/'manifests/c4_unseen_rules_v1_feedsllm_20260906.sha256').read_text().splitlines()}
    names=['evaluate_c4_unseen_rules.py','c4_unseen_rules.py','c4_execution_model.py',
           'c4_execution_data.py','c4_rule_followup.py','c4_freegen_protocol.py',
           'train_c1_lora.py','state_swap_misconception_smoke.py']
    fp={'code':{n:frozen['scripts/'+n] for n in names}}
    frozen_code_checks(fp,frozen)
    fp['code'][names[0]]='0'*64
    with pytest.raises(AssertionError):frozen_code_checks(fp,frozen)
    del fp['code'][names[0]]
    with pytest.raises(AssertionError):frozen_code_checks(fp,frozen)


def test_report_preserves_negative_categories_and_all_conditions():
    m=json.loads((ROOT/'manifests/c4_unseen_rules_v1_20260906.json').read_text())
    zero=dict(estimate=0,pair_cluster_95ci=[0,0],n_independent_pairs=4)
    agg=dict(arms={},fingerprints={},B_minus_A_by_category={c:zero for c in PAIRS})
    details={}
    for arm in ('A','B'):
        agg['fingerprints'][arm]={'adapter_sha256':m['checkpoints'][arm]['adapter_sha256']}
        agg['arms'][arm]={cat:dict(specificity=zero,noop_mismatches=[],
            counts={c:dict(n=24,donor_hits=0,invalid=0) for c in CONDITIONS},
            joint3={c:dict(n_groups=8,all3_donor_hits=0) for c in CONDITIONS}) for cat in PAIRS}
        gs=[]
        for g in m['groups']:
            qs=[dict(problem=q['problem_text'],expected_donor=q['targets']['donor'],
                outputs={c:q['targets']['recipient'] for c in CONDITIONS},
                texts={c:q['targets']['recipient'] for c in CONDITIONS},stop={c:'eos' for c in CONDITIONS}) for q in g['queries']]
            gs.append(dict(group_id=g['id'],category=g['category'],donor_rule=g['rules']['donor'],
                recipient_rule=g['rules']['recipient'],queries=qs))
        details[arm]=dict(by_rule={},groups=gs,output_formats={'bare':864},stop_reasons={'eos':864})
    text=render(dict(aggregate=agg,details=details,audit='synthetic audit',limitation='synthetic only'))
    for c in CONDITIONS:assert c in text
    for category in PAIRS:assert f'| A | {category} | 0/24 | 0/24 |' in text
    assert 'no all-three success group' in text
    assert 'spontaneous invention' in text and 'four independent' in text
    assert 'does not support reliable unseen-rule state transport' in text
    assert 'own-state no-op output mismatches 0/96' in text


def test_delivery_end_to_end_synthetic_artifacts(tmp_path):
    """Exercise actual collector/inspect across all files; outputs are synthetic."""
    from transformers import AutoTokenizer
    from audit_c4_unseen_rules_delivery import inspect,MODEL
    from c4_freegen_protocol import sha
    from c4_unseen_rules import prompt
    manifest=ROOT/'manifests/c4_unseen_rules_v1_20260906.json'
    m=json.loads(manifest.read_text())
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    frozen={line.split('  ',1)[1]:line.split('  ',1)[0] for line in
        (ROOT/'manifests/c4_unseen_rules_v1_feedsllm_20260906.sha256').read_text().splitlines()}
    names=['evaluate_c4_unseen_rules.py','c4_unseen_rules.py','c4_execution_model.py',
           'c4_execution_data.py','c4_rule_followup.py','c4_freegen_protocol.py',
           'train_c1_lora.py','state_swap_misconception_smoke.py']
    versions=dict(line.split('==') for line in (ROOT/'requirements-feeds.txt').read_text().splitlines())
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
    for arm in ('A','B'):
        root=tmp_path/f'c4-unseen-rules-arm-{arm.lower()}-seed42'/f'unseen_{arm}'
        root.mkdir(parents=True)
        fp=dict(arm=arm,manifest_sha256=sha(manifest),
            adapter_sha256=m['checkpoints'][arm]['adapter_sha256'],
            code={n:frozen['scripts/'+n] for n in names},
            base={n:sha(MODEL/n) for n in ('config.json','tokenizer.json','model.safetensors.index.json')})
        write(root/'lineage.json',fp)
        write(root.parent/f'runtime_{arm}.json',dict(fingerprint=fp,arm=arm,versions=versions))
        all_records=[]
        for g in m['groups']:
            rows=[]
            for q in g['queries']:
                for c in CONDITIONS:
                    y=q['targets']['donor'] if c in ('swap','donor_native','same_donor','oracle','open_history') else q['targets']['recipient']
                    text=f'<predicted_response>{y}</predicted_response>'
                    rows.append(dict(group_id=g['id'],pair_id=g['pair_id'],category=g['category'],
                        query_idx=q['query_idx'],condition=c,targets=q['targets'],correct=q['correct'],
                        prompt_sha256=g['prompt_hashes'][f'{q["query_idx"]}:{c}'],
                        prompt_tokens=len(tok(prompt(g,q,c).replace('<STUDENT_STATE>','<|vision_pad|>'),add_special_tokens=False)['input_ids']),
                        tokens=tok(text,add_special_tokens=False)['input_ids'],text=text,parsed=y,
                        parse_type='tagged',stop='closing_tag'))
            write(root/(g['id'].replace(':','_')+'.json'),
                dict(fingerprint=fp,state_hashes={role:'a'*64 for role in g['histories']},records=rows))
            all_records.extend(rows)
        write(root/'records.json',all_records)
        write(root/'complete.json',dict(status='complete',fingerprint=fp,
            records=len(all_records),records_sha256=sha(root/'records.json')))
    result=inspect(manifest=manifest,downloads=tmp_path,tokenizer=tok)
    assert result['status']=='complete' and result['aggregate']['n_records']==1728
    for arm in ('A','B'):
        assert result['details'][arm]['prompt_length_checks']==864
        for cat in PAIRS:
            assert result['aggregate']['arms'][arm][cat]['specificity']['estimate']==1
            assert result['aggregate']['arms'][arm][cat]['joint3']['swap']['all3_donor_hits']==8
    assert 'all 1,728 records passed' in render(result)
    assert 'does not support reliable unseen-rule state transport' not in render(result)
    missing=tmp_path/'c4-unseen-rules-arm-a-seed42/unseen_A'/ (m['groups'][0]['id'].replace(':','_')+'.json')
    missing.unlink()
    with pytest.raises(AssertionError):inspect(manifest=manifest,downloads=tmp_path,tokenizer=tok)
