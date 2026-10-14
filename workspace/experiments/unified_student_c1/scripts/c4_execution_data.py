"""Frozen small execution-training panel; no model-selected examples."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

from c4_freegen_protocol import numeric_tokens, signatures, sha
from c4_rule_followup import RULES, independent_answer

ROOT = Path(__file__).resolve().parents[1]
SUB = tuple('subtraction.'+s for s in ('smaller_from_larger','borrow_no_decrement','diff_0_n_equals_n','stops_borrow_at_zero'))
FRAC = ('fractions.ignore_denominators','fractions.keep_common_denominator_for_multiplication')
INSTRUCTION = ('You are modeling one student\'s mathematical behavior. '
               'Predict the exact answer this same student would give on the next problem. '
               'Return exactly one <predicted_response> tag.\n')
STATE = '<STUDENT_STATE>'

def digest_json(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def answer(rule, q):
    p=SimpleNamespace(operands=q['operands'], operation=q['operation'],metadata={})
    return independent_answer(rule,p)

def correct(q):
    if q['operation']=='subtraction': return str(q['operands'][0]-q['operands'][1])
    a,b,c,d=q['operands']; return str(Fraction(a*c,b*d))

def history(events, rule):
    lines=['<student_history>']
    for i,q in enumerate(events,1):
        lines += [f'Trial {i}:',f"Problem: {q['problem_text']}",f'Student answer: {answer(rule,q)}',
                  f'Correct answer: {correct(q)}','Outcome: incorrect','']
    return '\n'.join(lines).rstrip()+'\n</student_history>\n'

def prompt(context, q, rule=None, menu=None):
    h=context if rule is None else '<student_history>\n</student_history>\n'
    text=INSTRUCTION+h+STATE+'\n<query>\n'
    if rule is not None: text+='Student rule: Follow this rule even if mathematically incorrect: '+RULES[rule]+'\n'
    text+='Problem: '+q['problem_text']+'\n'
    if menu is not None: text+='Valid outputs:\n'+'\n'.join('<predicted_response>'+a+'</predicted_response>' for a in menu)+'\n'
    return text+'</query>\nAnswer:'

def rows(m,split):
    """Identical target sequence/exposure for both arms; first half explicit-rule."""
    result=[]
    for mode in ('oracle','history'):
        rs=[]
        for g in m['splits'][split]:
            for q in g['queries']:
                rs.append(dict(group_id=g['id'],query_idx=q['query_idx'],mode=mode,
                    donor_prefix=INSTRUCTION+g['histories']['donor']+STATE,
                    native=prompt(g['histories']['donor'],q,g['rules']['donor'] if mode=='oracle' else None),
                    recipient=prompt(g['histories']['recipient'],q),
                    target='<predicted_response>'+q['targets']['donor']+'</predicted_response>'))
        random.Random(m['seed']+(0 if mode=='oracle' else 1)).shuffle(rs)
        result.extend(rs)
    return result

def build(root=ROOT):
    from prepare_malrule_c1 import discover_malrules
    repo=root/'data/raw/malrulelib'
    infos={i.malrule_id:i for i in discover_malrules(repo,['subtraction','fractions'])}
    seen=set(); lineage=[]
    files=[root/'data/processed/state_curriculum_stage_a_v1'/s for s in ('train.jsonl','valid.jsonl')]
    files += [root/'data/processed/malrule_c1_clean'/s for s in ('train.jsonl','val.jsonl','test_same_template.jsonl','test_cross_template.jsonl')]
    for f in files:
        n=0
        for line in f.open():
            r=json.loads(line); n+=1
            for q in r.get('history',[])+[r['query']]:
                try: seen.update(signatures(q))
                except (ValueError,ZeroDivisionError): seen.add(('text',tuple(sorted(numeric_tokens(q['problem_text'])))))
        lineage.append(dict(path=str(f.relative_to(root)),sha256=sha(f),rows=n))
    old=root/'manifests/c4_freegen_novel_seed42_20260905.json'
    for g in json.loads(old.read_text())['groups']:
        for q in g['queries']: seen.update(signatures(q))
    lineage.append(dict(path=str(old.relative_to(root)),sha256=sha(old)))
    seed=2026090601; rng=random.Random(seed); rejects=Counter(); checked=0
    def sample(family,rule_set,forbidden=(),past_answers=()):
        nonlocal checked
        for attempt in range(200000):
            if family=='subtraction':
                a=rng.randrange(100,10000); b=rng.randrange(10,a)
                text=rng.choice((f'Calculate: {a} - {b}',f'Subtract {b} from {a}.',f'What is {a} minus {b}?'))
                q=dict(problem_text=text,operands=[a,b],operation='subtraction')
            else:
                d=rng.randrange(4,41); a=rng.randrange(1,d); c=rng.randrange(1,d)
                text=rng.choice((f'Multiply: {a}/{d} × {c}/{d}',f'What is {a}/{d} times {c}/{d}?',f'Calculate ({a}/{d}) * ({c}/{d}).'))
                q=dict(problem_text=text,operands=[a,d,c,d],operation='fraction_multiplication')
            sig=signatures(q)
            if seen & sig: rejects['seen_numbers']+=1; continue
            ys=[answer(r,q) for r in rule_set]
            if len(set(ys+[correct(q)]))!=len(ys)+1: rejects['answers_collide']+=1; continue
            if any(y in forbidden or y in past_answers for y in ys): rejects['copy_or_repeated_answer']+=1; continue
            # Independent implementation versus existing library where it derives
            # answers from operands. Keep-common library reads target metadata;
            # verify this rule with direct Fraction arithmetic instead (not a false
            # claim that target-carrying metadata independently validates itself).
            p=SimpleNamespace(text=q['problem_text'],operands=q['operands'],operation=q['operation'],metadata={})
            for rule,y in zip(rule_set,ys):
                if rule==FRAC[1]:
                    a,d,c,_=q['operands']; ref=str(Fraction(a*c,d))
                else: ref=str(Fraction(str(infos[rule].malrule_cls().calculate_expected_answer(p))))
                assert y==ref,(rule,q,y,ref)
                checked+=1
            seen.update(sig)
            return q
        raise RuntimeError(f'Cannot sample {family} {rule_set}; {rejects}')
    splits={}
    for split,n in (('train',80),('valid',4),('test',8)):
        groups=[]
        for fam,pool in (('subtraction',SUB),('fractions',FRAC)):
            for pair in range(n):
                ra,rb=rng.sample(pool,2)
                # Different histories, three diagnostic observations per history.
                histories={}; events={}; rulemap={'a':ra,'b':rb,'same_a':ra,'same_b':rb}
                for role,r in rulemap.items():
                    es=[sample(fam,(ra,rb)) for _ in range(3)]
                    events[role]=es; histories[role]=history(es,r)
                forbidden=set(numeric_tokens(' '.join(histories.values())))
                queries=[]; targets_seen=set()
                for qi in range(3):
                    q=sample(fam,(ra,rb),forbidden,targets_seen)
                    targets_seen.update((answer(ra,q),answer(rb,q)))
                    queries.append(dict(q,query_idx=qi))
                for reverse in (False,True):
                    donor,recipient=('b','a') if reverse else ('a','b')
                    same='same_'+donor; random_role='same_'+recipient
                    roles={'donor':donor,'recipient':recipient,'same_donor':same,'random':random_role}
                    g=dict(id=f'{split}:{fam}:{pair:03d}:{int(reverse)}',pair_id=f'{split}:{fam}:{pair:03d}',family=fam,
                        rules={k:rulemap[v] for k,v in roles.items()},histories={k:histories[v] for k,v in roles.items()},
                        history_events={k:events[v] for k,v in roles.items()},queries=[])
                    for q in queries:
                        g['queries'].append(dict(q,targets={k:answer(rulemap[v],q) for k,v in roles.items()},correct=correct(q)))
                    groups.append(g)
        splits[split]=groups
    m=dict(protocol='c4_execution_training_v1',seed=seed,splits=splits,rules={r:RULES[r] for r in SUB+FRAC},
        lineage=lineage,rejections=dict(rejects),independent_answer_checks=checked,
        init_adapter_sha256=json.loads(old.read_text())['adapter_sha256'],
        scope='Exploratory six-rule arithmetic repair, one training seed; new histories and number combinations, not unseen rules/templates or human data.',
        library_hashes={str(p.relative_to(root)):sha(p) for r in SUB+FRAC for p in [repo/'malrules'/r.replace('.','/')/'malrule_algorithm.py']},
        training=dict(seed=42,lr=2e-5,steps=240,accumulation=8,warmup_steps=12,max_length=768,dropout=0.0),
        evaluation=dict(max_new_tokens=48,conditions=['native','donor_native','noop','swap','random','same_donor','oracle','menu_native','menu_swap','menu_random']))
    assert len(rows(m,'train'))==1920
    m['row_counts']={s:len(rows(m,s)) for s in splits}
    m['history_identifiability']={s:{g['id']:{role:[r for r in (SUB if g['family']=='subtraction' else FRAC)
        if all(answer(r,q)==answer(g['rules'][role],q) for q in es)] for role,es in g['history_events'].items()}
        for g in gs} for s,gs in splits.items()}
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=build();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({k:m[k] for k in ('row_counts','rejections','independent_answer_checks','training')},indent=2))

if __name__=='__main__': main()
