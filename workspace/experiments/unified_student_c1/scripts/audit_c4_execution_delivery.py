"""Post-run integrity and descriptive rule/no-op audit; no model selection."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from collect_c4_execution_training import collect, ci
from c4_freegen_protocol import sha


def audit_prediction_lineage(root, label, manifest, manifest_hash, tokenizer=None):
    """Check every state group, token/text record and baseline identity."""
    fp=json.loads((root/'lineage.json').read_text())
    assert fp['label']==label
    assert fp['manifest_sha256']==manifest_hash
    assert fp['protocol']=='execution_v1_corrected_anchor'
    if label=='baseline':assert fp['adapter_sha256']==manifest['init_adapter_sha256']
    expected={g['id'].replace(':','_')+'.json' for g in manifest['splits']['test']}
    found={p.name for p in root.glob('test_*.json')}
    assert found==expected,(label,'missing or extra prediction group',found^expected)
    n=0
    for name in sorted(expected):
        group=json.loads((root/name).read_text())
        assert group['fingerprint']==fp,(label,name,'group fingerprint')
        states=group['state_hashes']
        assert set(states)=={'donor','recipient','random','same_donor'}
        for digest in states.values():
            assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
        for r in group['records']:
            assert r['group_id'].replace(':','_')+'.json'==name
            assert r['stop'] in {'token_limit','eos','closing_tag'}
            assert isinstance(r['tokens'],list) and all(isinstance(t,int) and t>=0 for t in r['tokens'])
            assert len(r['tokens'])<=manifest['evaluation']['max_new_tokens']
            if r['stop']=='token_limit':assert len(r['tokens'])==manifest['evaluation']['max_new_tokens']
            if r['stop']=='closing_tag':assert '</predicted_response>' in r['text']
            if tokenizer is not None:
                assert tokenizer.decode(r['tokens'],skip_special_tokens=False)==r['text'],(label,name,'token/text mismatch')
            n+=1
    assert n==len(manifest['splits']['test'])*3*len(manifest['evaluation']['conditions'])
    return dict(groups=len(found),records=n,token_text_checked=tokenizer is not None)


def audit_training(root, arm, manifest, manifest_hash):
    lineage=json.loads((root/'lineage.json').read_text())
    cfg=manifest['training']
    assert lineage['arm']==arm
    assert lineage['manifest_sha256']==manifest_hash
    assert lineage['initial_adapter_sha256']==manifest['init_adapter_sha256']
    assert lineage['training']==cfg
    logs=[json.loads(line) for line in (root/'train.jsonl').read_text().splitlines() if line.strip()]
    # A preempted invocation can repeat steps after its last complete checkpoint.
    # Keep all attempts; require each step and final completion, flag replays.
    steps=defaultdict(list)
    for row in logs:
        step=row['step'];assert isinstance(step,int) and 1<=step<=cfg['steps']
        assert row['phase']==('oracle' if step<=120 else 'history')
        for key in ('loss','grad_norm','lr','elapsed','peak_vram_gib'):
            assert math.isfinite(row[key]),(arm,step,key)
        steps[step].append(row)
    assert set(steps)==set(range(1,cfg['steps']+1)),(arm,'missing training steps')
    assert logs[-1]['step']==cfg['steps']
    validations={}
    for step in (120,240):
        data=json.loads((root/f'validation-{step}.json').read_text())
        assert set(data)=={'oracle','history'}
        for v in data.values():assert math.isfinite(v['loss']) and v['tokens']>0
        validations[step]=data
    done=json.loads((root/'complete.json').read_text())
    checkpoint=json.loads((root/f'checkpoint-{cfg["steps"]}/complete.json').read_text())
    assert checkpoint['step']==cfg['steps']
    assert checkpoint['adapter_sha256']==done['final_adapter_sha256']
    exposure=json.loads((root/'exposure.json').read_text())
    assert exposure['rows']==1920 and exposure['supervised_tokens']==26524
    assert exposure['target_order_sha256']=='a6d3a2049010a18cd7b376dd573f87aaf6b8b190ae3c035934fc42b1ce5ac558'
    return dict(logged_updates=len(logs),unique_steps=len(steps),
        replayed_steps={str(s):len(rs) for s,rs in steps.items() if len(rs)>1},
        first_update=logs[0],last_update=logs[-1],validation=validations,
        source_code=lineage['code'],initial_adapter_sha256=lineage['initial_adapter_sha256'])


def audit(manifest_path, downloads, tokenizer=None):
    manifest=json.loads(manifest_path.read_text())
    core=collect(manifest_path,downloads)
    training={};models={};examples={}
    for arm in ('A','B'):
        root=next(downloads.glob(f'*/execution_{arm}'))
        training[arm]=audit_training(root,arm,manifest,sha(manifest_path))
        for label,subdir in ([('baseline','baseline'),('A','final')] if arm=='A' else [('B','final')]):
            integrity=audit_prediction_lineage(root/subdir,label,manifest,sha(manifest_path),tokenizer)
            records=json.loads((root/subdir/'records.json').read_text())
            idx={(r['group_id'],r['query_idx'],r['condition']):r for r in records}
            rule_counts=defaultdict(lambda:defaultdict(lambda:dict(n=0,donor_hits=0,recipient_hits=0,invalid=0)))
            joint=defaultdict(lambda:dict(n_groups=0,all3_donor_hits=0))
            noop=[];pair_contrasts=defaultdict(list);candidate_contrasts=defaultdict(list)
            successes=[];failures=[]
            for group in manifest['splits']['test']:
                rule=group['rules']['donor'];hits={}
                for cond in manifest['evaluation']['conditions']:
                    rs=[idx[(group['id'],q['query_idx'],cond)] for q in group['queries']]
                    hits[cond]=[r['parsed']==r['targets']['donor'] for r in rs]
                    joint[cond]['n_groups']+=1
                    joint[cond]['all3_donor_hits']+=int(all(hits[cond]))
                    for r in rs:
                        count=rule_counts[rule][cond];count['n']+=1
                        count['donor_hits']+=int(r['parsed']==r['targets']['donor'])
                        count['recipient_hits']+=int(r['parsed']==r['targets']['recipient'])
                        count['invalid']+=int(r['parsed'] is None)
                for q in group['queries']:
                    native=idx[(group['id'],q['query_idx'],'native')]
                    no_op=idx[(group['id'],q['query_idx'],'noop')]
                    if native['tokens']!=no_op['tokens'] or native['stop']!=no_op['stop']:
                        noop.append(dict(group_id=group['id'],query_idx=q['query_idx'],
                            native=native['text'],noop=no_op['text'],
                            native_donor_hit=native['parsed']==q['targets']['donor'],
                            noop_donor_hit=no_op['parsed']==q['targets']['donor'],
                            native_recipient_hit=native['parsed']==q['targets']['recipient'],
                            noop_recipient_hit=no_op['parsed']==q['targets']['recipient']))
                pair_contrasts[group['pair_id']].extend(int(a)-int(b) for a,b in zip(hits['swap'],hits['random']))
                candidate_contrasts[group['pair_id']].extend(int(a)-int(b) for a,b in zip(hits['menu_swap'],hits['menu_random']))
                example=dict(group_id=group['id'],pair_id=group['pair_id'],donor_rule=rule,
                    donor_history=group['histories']['donor'],recipient_history=group['histories']['recipient'],
                    queries=[dict(problem=q['problem_text'],targets=q['targets'],
                        swap=idx[(group['id'],q['query_idx'],'swap')]['text'],
                        random=idx[(group['id'],q['query_idx'],'random')]['text']) for q in group['queries']])
                # Deterministic illustrative examples only, never sample filtering.
                if all(hits['swap']) and not any(hits['random']):successes.append(example)
                elif not all(hits['swap']):failures.append(example)
            models[label]=dict(integrity=integrity,noop_token_mismatches=noop,noop_n_queries=96,
                per_donor_rule=dict(rule_counts),all_three_by_condition=dict(joint),
                open_specificity=ci(pair_contrasts),candidate_visible_specificity=ci(candidate_contrasts))
            examples[label]=dict(first_success=successes[:1],first_failure=failures[:1])
    assert training['A']['source_code']==training['B']['source_code']
    return dict(status='complete',core=core,training=training,models=models,examples=examples,
        caveat='One exploratory seed; 16 independent history pairs; six known rules. Candidate-visible controls are not free generation. Any no-op discrepancy or replay must be reported, not silently removed.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','downloads','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,help='Optional local tokenizer for raw token/text integrity')
    a=p.parse_args();tokenizer=None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    result=audit(a.manifest,a.downloads,tokenizer)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],B_minus_A=result['core']['B_minus_A_specificity']),indent=2))
