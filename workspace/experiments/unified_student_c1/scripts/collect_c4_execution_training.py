"""Recompute paired endpoints on frozen test; do not select a checkpoint."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from c4_execution_data import prompt
from c4_freegen_protocol import parse_output,sha

def ci(values,n=10000):
    vals=list(values.values());rng=random.Random(2026090602)
    avg=lambda xs:sum(sum(v) for v in xs)/sum(map(len,xs))
    draws=sorted(avg([rng.choice(vals) for _ in vals]) for _ in range(n))
    return dict(estimate=avg(vals),pair_cluster_95ci=[draws[int(n*.025)],draws[int(n*.975)]],n_independent_pairs=len(vals))

def audit_eval(m,root,manifest_hash):
    records=json.loads((root/'records.json').read_text());fp=json.loads((root/'lineage.json').read_text())
    assert fp['manifest_sha256']==manifest_hash
    lookup={g['id']:g for g in m['splits']['test']}
    idx={}
    for r in records:
        key=(r['group_id'],r['query_idx'],r['condition']);assert key not in idx
        idx[key]=r;g=lookup[r['group_id']];q=g['queries'][r['query_idx']]
        assert r['targets']==q['targets'] and r['pair_id']==g['pair_id']
        parsed,kind=parse_output(r['text'])
        if r['stop']=='token_limit':parsed=None;kind='truncated'
        assert parsed==r['parsed'] and kind==r['parse_type']
        c=r['condition'];menu=None
        if c.startswith('menu_'):
            menu=sorted(set([q['targets']['donor'],q['targets']['recipient'],q['correct']]))
            random.Random(g['id']+str(q['query_idx'])).shuffle(menu)
        p=prompt(g['histories']['donor' if c=='donor_native' else 'recipient'],q,g['rules']['donor'] if c=='oracle' else None,menu)
        import hashlib
        assert hashlib.sha256(p.encode()).hexdigest()==r['prompt_sha256']
    expected={(g['id'],q['query_idx'],c) for g in lookup.values() for q in g['queries'] for c in m['evaluation']['conditions']}
    assert set(idx)==expected and len(records)==len(expected)
    groups=[]
    for f in root.glob('*_subtraction_*.json'):groups+=json.loads(f.read_text())['records']
    for f in root.glob('*_fractions_*.json'):groups+=json.loads(f.read_text())['records']
    assert sorted(groups,key=lambda r:(r['group_id'],r['query_idx'],r['condition']))==sorted(records,key=lambda r:(r['group_id'],r['query_idx'],r['condition']))
    counts={}
    for c in m['evaluation']['conditions']:
        rs=[r for r in records if r['condition']==c]
        counts[c]=dict(n=len(rs),donor_hits=sum(r['parsed']==r['targets']['donor'] for r in rs),
            recipient_hits=sum(r['parsed']==r['targets']['recipient'] for r in rs),invalid=sum(r['parsed'] is None for r in rs))
    assert counts==json.loads((root/'summary.json').read_text())
    contrasts={}
    for endpoint in ('individual','joint3'):
        values=defaultdict(list)
        for g in lookup.values():
            hit=lambda q,c:idx[(g['id'],q['query_idx'],c)]['parsed']==q['targets']['donor']
            if endpoint=='individual':values[g['pair_id']].extend(int(hit(q,'swap'))-int(hit(q,'random')) for q in g['queries'])
            else:values[g['pair_id']].append(int(all(hit(q,'swap') for q in g['queries']))-int(all(hit(q,'random') for q in g['queries'])))
        contrasts[endpoint]=ci(values)
    return dict(counts=counts,contrasts=contrasts,records=records,sha256=sha(root/'records.json'),fingerprint=fp)

def collect(manifest,downloads):
    m=json.loads(manifest.read_text());arms={}
    for arm in ('A','B'):
        candidates=list(downloads.glob(f'*/execution_{arm}'))
        assert len(candidates)==1,(arm,candidates)
        root=candidates[0];done=json.loads((root/'complete.json').read_text())
        assert done['status']=='complete' and done['steps']==m['training']['steps']
        arms[arm]=audit_eval(m,root/'final',sha(manifest))
        assert done['final_adapter_sha256']==arms[arm]['fingerprint']['adapter_sha256']
        arms[arm]['exposure']=json.loads((root/'exposure.json').read_text())
        if arm=='A':arms['baseline']=audit_eval(m,root/'baseline',sha(manifest))
    for k in ('rows','supervised_tokens','target_order_sha256'):
        assert arms['A']['exposure'][k]==arms['B']['exposure'][k],k
    idx={arm:{(r['group_id'],r['query_idx'],r['condition']):r for r in a['records']} for arm,a in arms.items()}
    contrasts={}
    for lhs,rhs in (('B','A'),('A','baseline'),('B','baseline')):
        values=defaultdict(list)
        for g in m['splits']['test']:
            for q in g['queries']:
                def hit(arm,c):return int(idx[arm][(g['id'],q['query_idx'],c)]['parsed']==q['targets']['donor'])
                values[g['pair_id']].append(hit(lhs,'swap')-hit(lhs,'random')-hit(rhs,'swap')+hit(rhs,'random'))
        contrasts[lhs+'_minus_'+rhs+'_specificity']=ci(values)
    return dict(status='complete',**contrasts,arms={arm:{k:v for k,v in a.items() if k!='records'} for arm,a in arms.items()},
        scope='Exploratory pilot; pair-cluster intervals, one checkpoint and six known rules. Main endpoint is per-query specificity B minus A. Joint3 and all other intervals secondary, unadjusted.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--downloads',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=collect(a.manifest,a.downloads);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
