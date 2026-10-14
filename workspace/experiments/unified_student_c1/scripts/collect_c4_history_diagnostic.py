"""Fail-closed merge and paired history-length contrasts; no output exclusions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from c4_history_diagnostic import COHORTS, CONDITIONS, validate
from c4_unseen_rules import PAIRS, RULES, answer, digest, prompt
from c4_freegen_protocol import parse_output, sha
from collect_c4_unseen_rules import interval


def audit_records(m, records, tokenizer, cohort):
    groups={g['id']:g for g in m['groups'] if g['cohort']==cohort}
    expected={(g['id'],q['query_idx'],c) for g in groups.values() for q in g['queries'] for c in CONDITIONS}
    idx={}
    for r in records:
        key=(r['group_id'],r['query_idx'],r['condition'])
        assert key in expected and key not in idx
        idx[key]=r;g=groups[key[0]];q=g['queries'][key[1]]
        assert r['cohort']==cohort and r['original_group_id']==g['original_group_id']
        assert r['pair_id']==g['pair_id'] and r['category']==g['category']
        assert r['targets']==q['targets'] and r['correct']==q['correct']
        text=prompt(g,q,key[2])
        assert r['prompt_sha256']==g['prompt_hashes'][f'{key[1]}:{key[2]}']==digest(text)
        length=len(tokenizer(text.replace('<STUDENT_STATE>','<|vision_pad|>'),add_special_tokens=False)['input_ids'])
        assert r['prompt_tokens']==length and length+m['max_new_tokens']<=m['max_length']
        assert r['stop'] in ('closing_tag','eos','token_limit')
        assert len(r['tokens'])<=m['max_new_tokens']
        assert all(isinstance(t,int) and t>=0 for t in r['tokens'])
        assert tokenizer.decode(r['tokens'],skip_special_tokens=False)==r['text']
        parsed,kind=parse_output(r['text'])
        if r['stop']=='token_limit':
            assert len(r['tokens'])==m['max_new_tokens'];parsed,kind=None,'truncated'
        if r['stop']=='closing_tag':assert '</predicted_response>' in r['text']
        assert (r['parsed'],r['parse_type'])==(parsed,kind)
    assert set(idx)==expected and len(records)==m['n_generations_per_shard']
    return idx


def summarize(m, indices, draws=10000):
    result={};contrasts={};examples={};oracle_invariance={}
    lookup={(g['cohort'],g['original_group_id']):g for g in m['groups']}
    for arm,idx in indices.items():
        result[arm]={};contrasts[arm]={};examples[arm]={};oracle_invariance[arm]=[]
        for category in PAIRS:
            base=[g for g in m['groups'] if g['cohort']=='h3' and g['category']==category]
            result[arm][category]={};contrasts[arm][category]={}
            def row(g,q,c,cohort):return idx[(lookup[cohort,g['original_group_id']]['id'],q['query_idx'],c)]
            def hit(g,q,c,cohort):return int(row(g,q,c,cohort)['parsed']==q['targets']['donor'])
            for g in base:
                for q in g['queries']:
                    ref=row(g,q,'oracle','h3')
                    for cohort in COHORTS[1:]:
                        obs=row(g,q,'oracle',cohort)
                        if (ref['tokens'],ref['stop'])!=(obs['tokens'],obs['stop']):
                            oracle_invariance[arm].append(dict(original_group_id=g['original_group_id'],
                                query_idx=q['query_idx'],cohort=cohort,h3=ref['text'],observed=obs['text']))
            for cohort in COHORTS:
                counts={};joint={};noop=[];values=defaultdict(list);snap={}
                for c in CONDITIONS:
                    rs=[row(g,q,c,cohort) for g in base for q in g['queries']]
                    counts[c]=dict(n=len(rs),donor_hits=sum(r['parsed']==r['targets']['donor'] for r in rs),
                        recipient_hits=sum(r['parsed']==r['targets']['recipient'] for r in rs),invalid=sum(r['parsed'] is None for r in rs))
                    joint[c]=dict(n_groups=len(base),all3_donor_hits=sum(all(hit(g,q,c,cohort) for q in g['queries']) for g in base))
                    # Whole three-query signature avoids mistaking conditional half-hits for a rule.
                    labels=Counter()
                    for g in base:
                        compatible=[rule for rule in RULES if all(row(g,q,c,cohort)['parsed']==answer(rule,q) for q in g['queries'])]
                        labels['|'.join(compatible) if compatible else 'other_or_inconsistent']+=1
                    snap[c]=dict(labels)
                for g in base:
                    values[g['pair_id']].extend(hit(g,q,'swap',cohort)-hit(g,q,'random',cohort) for q in g['queries'])
                    for q in g['queries']:
                        x=row(g,q,'native',cohort);y=row(g,q,'noop',cohort)
                        if (x['tokens'],x['stop'])!=(y['tokens'],y['stop']):
                            noop.append(dict(group_id=x['group_id'],query_idx=q['query_idx'],native=x['text'],noop=y['text']))
                result[arm][category][cohort]=dict(counts=counts,joint3=joint,
                    specificity=interval(values,draws),noop_mismatches=noop,whole_triplet_rule_signatures=snap)
            for comparison,lo in (('h12_minus_h3','h3'),('h12_minus_repeat3x4','repeat3x4'),('h6_minus_h3','h3')):
                hi='h6' if comparison.startswith('h6') else 'h12'
                measures={}
                for measure in ('open_history','donor_native','swap','specificity'):
                    values=defaultdict(list);up=down=0
                    for g in base:
                        for q in g['queries']:
                            def score(cohort):
                                return (hit(g,q,'swap',cohort)-hit(g,q,'random',cohort) if measure=='specificity' else hit(g,q,measure,cohort))
                            delta=score(hi)-score(lo);values[g['pair_id']].append(delta)
                            up+=int(delta>0);down+=int(delta<0)
                    measures[measure]=dict(**interval(values,draws),improved_queries=up,worsened_queries=down)
                contrasts[arm][category][comparison]=measures
            g=base[0]
            examples[arm][category]=dict(group_id=g['original_group_id'],donor_rule=g['rules']['donor'],
                selection='First original group, irrespective of outcome',queries=[dict(
                    problem=q['problem_text'],donor_target=q['targets']['donor'],
                    outputs={cohort:{c:row(g,q,c,cohort)['text'] for c in CONDITIONS} for cohort in COHORTS}) for q in g['queries']])
    return dict(arms=result,paired_history_contrasts=contrasts,examples=examples,
        identical_oracle_prompt_mismatches=oracle_invariance,
        scope='Post-hoc exploratory, four independent pairs/category, one training seed. Same queries as prior test. No new replication gate or multiple-comparison correction. Zero-width empirical intervals are not zero population uncertainty.')


def collect(manifest, root, tokenizer, draws=10000):
    m=json.loads(manifest.read_text());validate(m);indices={};fps={};versions=None;code=None;base=None
    for arm in ('A','B'):
        idx={}
        for cohort in COHORTS:
            folder=root/f'c4-history-{arm.lower()}-{cohort}'/'diagnostic'
            fp=json.loads((folder/'lineage.json').read_text())
            assert fp['arm']==arm and fp['cohort']==cohort and fp['manifest_sha256']==sha(manifest)
            assert fp['adapter_sha256']==m['checkpoints'][arm]['adapter_sha256']
            if base is None:base=fp['base']
            assert fp['base']==base
            if code is None:code=fp['code']
            assert code==fp['code']
            assert all(sha(Path(__file__).parent/n)==v for n,v in code.items())
            runtime=json.loads((folder/'runtime.json').read_text())
            assert runtime['fingerprint']==fp
            if versions is None:versions=runtime['versions']
            assert runtime['versions']==versions
            done=json.loads((folder/'complete.json').read_text())
            assert done['status']=='complete' and done['fingerprint']==fp
            assert done['records']==m['n_generations_per_shard'] and done['records_sha256']==sha(folder/'records.json')
            records=json.loads((folder/'records.json').read_text());gathered=[]
            gs=[g for g in m['groups'] if g['cohort']==cohort]
            expected_files={g['id'].replace(':','_')+'.json' for g in gs}|{'lineage.json','complete.json','runtime.json','records.json'}
            assert {p.name for p in folder.glob('*.json')}==expected_files
            for g in gs:
                piece=json.loads((folder/(g['id'].replace(':','_')+'.json')).read_text())
                assert piece['fingerprint']==fp and set(piece['state_hashes'])=={'donor','recipient','random'}
                assert all(len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in piece['state_hashes'].values())
                assert all(r['group_id']==g['id'] for r in piece['records'])
                gathered.extend(piece['records'])
            assert records==gathered
            idx.update(audit_records(m,records,tokenizer,cohort));fps[f'{arm}/{cohort}']=fp
        assert len(idx)==m['n_generations_per_checkpoint'];indices[arm]=idx
    return dict(status='complete',manifest_sha256=sha(manifest),n_records=sum(map(len,indices.values())),
        fingerprints=fps,versions=versions,**summarize(m,indices,draws))


def render(summary):
    lines=['# C4 history-length diagnostic: audited results','',summary['scope'],'',
        'All source design, interventions, sample construction and exclusions are in the frozen protocol. No generations are excluded.', '',
        '| Arm | Category | History | Open history | Donor native | Swap | Random | D | All-three swap | No-op mismatches |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,categories in summary['arms'].items():
        for category,cohorts in categories.items():
            for cohort,r in cohorts.items():
                cells=[f"{r['counts'][c]['donor_hits']}/{r['counts'][c]['n']}" for c in ('open_history','donor_native','swap','random')]
                lines.append('| '+' | '.join([arm,category,cohort]+cells+[f"{r['specificity']['estimate']:.4f}",f"{r['joint3']['swap']['all3_donor_hits']}/8",str(len(r['noop_mismatches']))])+' |')
    lines+=['','## Paired diagnostic contrasts','',
        '| Arm | Category | Contrast | Measure | Difference [95% pair-cluster CI] |','|---|---|---|---|---|']
    for arm,cats in summary['paired_history_contrasts'].items():
        for cat,comparisons in cats.items():
            for comp,measures in comparisons.items():
                for measure,r in measures.items():
                    lo,hi=r['pair_cluster_95ci']
                    lines.append(f"| {arm} | {cat} | {comp} | {measure} | {r['estimate']:.4f} [{lo:.4f}, {hi:.4f}] |")
    lines+=['','## Interpretation boundary','',
        'More unique history helping beyond repeated history supports an information-dose explanation in this panel. Oracle success with history failure is consistent with an induction limitation, not a uniquely established cause. Donor-native success with swap failure implicates transport; oracle failure leaves arithmetic or instruction-following unresolved. Repetition is not an exact token-length control. The longer context also shifts the state position and is outside the three-example repair-training format. Known controls and all no-op mismatches remain visible.', '',
        'Full per-condition counts, invalids, whole-triplet rule signatures, paired transitions, raw example text and provenance are retained in the summary JSON.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','root','tokenizer','output','report'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from transformers import AutoTokenizer
    s=collect(a.manifest,a.root,AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True))
    a.output.write_text(json.dumps(s,indent=2)+'\n');a.report.write_text(render(s))
    print(json.dumps(dict(status=s['status'],n_records=s['n_records'],summary=str(a.output))))
