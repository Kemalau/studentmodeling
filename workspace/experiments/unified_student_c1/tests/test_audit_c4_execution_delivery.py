"""Synthetic training receipts test the auditor, not model performance."""
import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_c4_execution_delivery import audit_training,audit_prediction_lineage


def test_training_audit_requires_steps_and_finite_metrics(tmp_path):
    m=dict(training=dict(steps=240),init_adapter_sha256='initial')
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    write('lineage.json',dict(arm='A',manifest_sha256='manifest',
        initial_adapter_sha256='initial',training=m['training'],code={'runner':'frozen'}))
    rows=[dict(step=s,phase='oracle' if s<=120 else 'history',loss=.3,
        grad_norm=.2,lr=.00002,elapsed=s,peak_vram_gib=30) for s in range(1,241)]
    def write_rows(values):
        (tmp_path/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    for s in (120,240):
        write(f'validation-{s}.json',dict(oracle=dict(loss=.2,tokens=10),history=dict(loss=.3,tokens=10)))
    write('complete.json',dict(final_adapter_sha256='final'))
    write('checkpoint-240/complete.json',dict(step=240,adapter_sha256='final'))
    write('exposure.json',dict(rows=1920,supervised_tokens=26524,
        target_order_sha256='a6d3a2049010a18cd7b376dd573f87aaf6b8b190ae3c035934fc42b1ce5ac558'))
    write_rows(rows)
    result=audit_training(tmp_path,'A',m,'manifest')
    assert result['unique_steps']==240 and result['replayed_steps']=={}
    write_rows(rows[:45]+rows[40:])
    assert audit_training(tmp_path,'A',m,'manifest')['replayed_steps']['41']==2
    write_rows(rows[1:])
    with pytest.raises(AssertionError):audit_training(tmp_path,'A',m,'manifest')
    bad=copy.deepcopy(rows);bad[3]['loss']=float('nan');write_rows(bad)
    with pytest.raises(AssertionError):audit_training(tmp_path,'A',m,'manifest')


def test_prediction_lineage_rejects_baseline_and_token_corruption(tmp_path):
    m=dict(init_adapter_sha256='initial',splits=dict(test=[dict(id='test:example:0')]),
        evaluation=dict(max_new_tokens=48,conditions=['swap']))
    fp=dict(label='baseline',manifest_sha256='manifest',adapter_sha256='initial',
        protocol='execution_v1_corrected_anchor')
    rows=[dict(group_id='test:example:0',tokens=[5],text='5',stop='eos') for _ in range(3)]
    group=dict(fingerprint=fp,state_hashes={r:'0'*64 for r in ('donor','recipient','random','same_donor')},records=rows)
    class Tokenizer:
        def decode(self,tokens,skip_special_tokens=False):return ''.join(map(str,tokens))
    def write():
        (tmp_path/'lineage.json').write_text(json.dumps(fp))
        (tmp_path/'test_example_0.json').write_text(json.dumps(group))
    write();result=audit_prediction_lineage(tmp_path,'baseline',m,'manifest',Tokenizer())
    assert result==dict(groups=1,records=3,token_text_checked=True)
    rows[0]['text']='6';write()
    with pytest.raises(AssertionError):audit_prediction_lineage(tmp_path,'baseline',m,'manifest',Tokenizer())
    rows[0]['text']='5';fp['adapter_sha256']='wrong';write()
    with pytest.raises(AssertionError):audit_prediction_lineage(tmp_path,'baseline',m,'manifest',Tokenizer())
