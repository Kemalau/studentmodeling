"""Audit all frozen unseen-rule outputs; paired descriptive statistics only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from c4_freegen_protocol import parse_output, sha
from c4_unseen_rules import CONDITIONS, PAIRS, digest, prompt


def interval(values, draws=10000):
    rng=random.Random(2026090604); units=list(values.values())
    mean=lambda rows:sum(map(sum,rows))/sum(map(len,rows))
    samples=sorted(mean([rng.choice(units) for _ in units]) for _ in range(draws))
    return dict(estimate=mean(units),pair_cluster_95ci=[samples[int(.025*draws)],samples[int(.975*draws)]],
                n_independent_pairs=len(units))


def audit_records(m, records, tokenizer):
    expected={(g['id'],q['query_idx'],c) for g in m['groups'] for q in g['queries'] for c in CONDITIONS}
    idx={}
    groups={g['id']:g for g in m['groups']}
    for r in records:
        k=(r['group_id'],r['query_idx'],r['condition']); assert k not in idx
        assert k in expected; idx[k]=r; g=groups[k[0]]; q=g['queries'][k[1]]
        assert r['pair_id']==g['pair_id'] and r['category']==g['category']
        assert r['targets']==q['targets'] and r['correct']==q['correct']
        assert r['prompt_sha256']==g['prompt_hashes'][f'{k[1]}:{k[2]}']
        assert r['prompt_sha256']==digest(prompt(g,q,k[2]))
        assert r['stop'] in ('eos','closing_tag','token_limit')
        assert len(r['tokens'])<=m['max_new_tokens']
        assert all(isinstance(t,int) and t>=0 for t in r['tokens'])
        assert tokenizer.decode(r['tokens'],skip_special_tokens=False)==r['text']
        parsed,kind=parse_output(r['text'])
        if r['stop']=='token_limit':
            assert len(r['tokens'])==m['max_new_tokens']; parsed,kind=None,'truncated'
        if r['stop']=='closing_tag': assert '</predicted_response>' in r['text']
        assert (r['parsed'],r['parse_type'])==(parsed,kind)
    assert set(idx)==expected and len(records)==m['n_generations_per_checkpoint']
    return idx


def summarize(m, indices, draws=10000):
    result={}; contrasts={}
    for arm,idx in indices.items():
        cats={}
        for category in PAIRS:
            gs=[g for g in m['groups'] if g['category']==category]
            counts={};joint={};noop=[];values=defaultdict(list)
            def hit(g,q,c):return int(idx[(g['id'],q['query_idx'],c)]['parsed']==q['targets']['donor'])
            for c in CONDITIONS:
                rs=[idx[(g['id'],q['query_idx'],c)] for g in gs for q in g['queries']]
                counts[c]=dict(n=len(rs),donor_hits=sum(r['parsed']==r['targets']['donor'] for r in rs),
                    recipient_hits=sum(r['parsed']==r['targets']['recipient'] for r in rs),invalid=sum(r['parsed'] is None for r in rs))
                joint[c]=dict(n_groups=len(gs),all3_donor_hits=sum(all(hit(g,q,c) for q in g['queries']) for g in gs))
            for g in gs:
                values[g['pair_id']].extend(hit(g,q,'swap')-hit(g,q,'random') for q in g['queries'])
                for q in g['queries']:
                    a=idx[(g['id'],q['query_idx'],'native')]; b=idx[(g['id'],q['query_idx'],'noop')]
                    if a['tokens']!=b['tokens'] or a['stop']!=b['stop']:
                        noop.append(dict(group_id=g['id'],query_idx=q['query_idx'],native=a['text'],noop=b['text']))
            cats[category]=dict(counts=counts,joint3=joint,specificity=interval(values,draws),noop_mismatches=noop)
        result[arm]=cats
    for category in PAIRS:
        values=defaultdict(list)
        for g in m['groups']:
            if g['category']!=category:continue
            for q in g['queries']:
                hit=lambda arm,c:int(indices[arm][(g['id'],q['query_idx'],c)]['parsed']==q['targets']['donor'])
                values[g['pair_id']].append(hit('B','swap')-hit('B','random')-hit('A','swap')+hit('A','random'))
        contrasts[category]=interval(values,draws)
    return dict(arms=result,B_minus_A_by_category=contrasts,
        scope='Exploratory; four independent pairs per category. No pooled new+seen success claim, no multiple-comparison correction or replication gate. Conditional-rule claims require joint-three evidence.')


def collect(manifest, roots, tokenizer):
    m=json.loads(manifest.read_text()); indices={};fps={}
    for arm,root in roots.items():
        fp=json.loads((root/'lineage.json').read_text()); fps[arm]=fp
        assert fp['arm']==arm and fp['manifest_sha256']==sha(manifest)
        assert fp['adapter_sha256']==m['checkpoints'][arm]['adapter_sha256']
        done=json.loads((root/'complete.json').read_text())
        assert done['status']=='complete' and done['fingerprint']==fp
        assert done['records_sha256']==sha(root/'records.json')
        records=json.loads((root/'records.json').read_text())
        assert done['records']==len(records)
        gathered=[]
        expected_group_files={g['id'].replace(':','_')+'.json' for g in m['groups']}
        actual_group_files={f.name for f in root.glob('*.json')} - {'lineage.json','complete.json','records.json'}
        assert actual_group_files==expected_group_files
        for g in m['groups']:
            group=json.loads((root/(g['id'].replace(':','_')+'.json')).read_text())
            assert group['fingerprint']==fp and set(group['state_hashes'])==set(g['histories'])
            assert all(len(s)==64 and all(c in '0123456789abcdef' for c in s) for s in group['state_hashes'].values())
            assert all(r['group_id']==g['id'] for r in group['records'])
            gathered.extend(group['records'])
        assert gathered==records
        indices[arm]=audit_records(m,records,tokenizer)
    assert set(roots)=={'A','B'}
    assert fps['A']['code']==fps['B']['code'] and fps['A']['base']==fps['B']['base']
    return dict(status='complete',manifest_sha256=sha(manifest),fingerprints=fps,
                n_records=sum(map(len,indices.values())),**summarize(m,indices))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','a','b','tokenizer','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    out=collect(args.manifest,dict(A=args.a,B=args.b),tok)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],n_records=out['n_records'],B_minus_A=out['B_minus_A_by_category']),indent=2))
