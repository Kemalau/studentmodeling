"""Collector tests use synthetic predictions, never model-result evidence."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from c4_execution_data import prompt
from c4_freegen_protocol import sha
from collect_c4_execution_training import collect

def test_matched_collection_rejects_corruption(tmp_path):
    manifest=ROOT/'manifests/c4_execution_training_v1_20260906.json'
    m=json.loads(manifest.read_text());downloads=tmp_path/'downloads'
    def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))
    for arm in ('A','B'):
        root=downloads/arm/f'execution_{arm}'
        write(root/'complete.json',dict(status='complete',steps=240,final_adapter_sha256='fake_'+arm))
        write(root/'exposure.json',dict(rows=1920,supervised_tokens=100,target_order_sha256='synthetic'))
        for model in (('baseline','final') if arm=='A' else ('final',)):
            out=root/model;records=[]
            for g in m['splits']['test']:
                rs=[]
                for q in g['queries']:
                    for c in m['evaluation']['conditions']:
                        y=q['targets']['donor'] if arm=='B' and c=='swap' else q['targets']['recipient']
                        menu=None
                        if c.startswith('menu_'):
                            menu=sorted(set([q['targets']['donor'],q['targets']['recipient'],q['correct']]))
                            random.Random(g['id']+str(q['query_idx'])).shuffle(menu)
                        p=prompt(g['histories']['donor' if c=='donor_native' else 'recipient'],q,g['rules']['donor'] if c=='oracle' else None,menu)
                        rs.append(dict(group_id=g['id'],pair_id=g['pair_id'],query_idx=q['query_idx'],condition=c,
                            targets=q['targets'],correct=q['correct'],text=f'<predicted_response>{y}</predicted_response>',
                            parsed=y,parse_type='tagged',stop='closing_tag',prompt_sha256=hashlib.sha256(p.encode()).hexdigest()))
                records.extend(rs);write(out/(g['id'].replace(':','_')+'.json'),dict(records=rs))
            write(out/'lineage.json',dict(manifest_sha256=sha(manifest),adapter_sha256='fake_'+arm))
            write(out/'records.json',records)
            write(out/'summary.json',{c:dict(n=96,donor_hits=sum(r['parsed']==r['targets']['donor'] for r in records if r['condition']==c),
                recipient_hits=sum(r['parsed']==r['targets']['recipient'] for r in records if r['condition']==c),invalid=0) for c in m['evaluation']['conditions']})
    result=collect(manifest,downloads)
    assert result['B_minus_A_specificity']['estimate']==1
    assert result['B_minus_A_specificity']['n_independent_pairs']==16
    assert result['A_minus_baseline_specificity']['estimate']==0
    assert result['B_minus_baseline_specificity']['estimate']==1
    assert result['A_minus_baseline_specificity']['pair_cluster_95ci']==[0,0]
    bad=downloads/'B/execution_B/final/records.json'
    r=json.loads(bad.read_text());write(bad,r+r[:1])
    with pytest.raises(AssertionError):collect(manifest,downloads)
