"""Paired, outcome-free history-length diagnostic of frozen C4 A/B models."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random

from c4_freegen_protocol import numeric_tokens, signatures, sha
from c4_unseen_rules import (ROOT, RULES, answer, correct, digest, history,
                             old_sources, prompt, reference)

SOURCE = 'manifests/c4_unseen_rules_v1_20260906.json'
MANIFEST = 'manifests/c4_history_diagnostic_v1_20260907.json'
COHORTS = ('h3', 'h6', 'h12', 'repeat3x4')
CONDITIONS = ('native', 'donor_native', 'noop', 'swap', 'random', 'oracle', 'open_history')
SEED = 2026090701


def diagnostic_history(events, rule):
    # Retain the old first three labels exactly. Letter labels avoid leaking a
    # small query target merely by numbering additional trials 4 through 12.
    text = history(events, rule)
    for i in range(4, len(events)+1):
        text = text.replace(f'Trial {i}:', f'Trial {chr(64+i)}:')
    return text


def build(root=ROOT):
    old = json.loads((root/SOURCE).read_text())
    seen, lineage = old_sources(root)
    for g in old['groups']:
        for es in g['history_events'].values():
            for q in es:
                seen.update(signatures(q))
        for q in g['queries']:
            seen.update(signatures(q))
    rng = random.Random(SEED)
    rejections = Counter()
    extended = {}
    # Work once per unordered pair; the reverse direction reuses the exact histories.
    for g in old['groups'][::2]:
        target_strings = {v for q in g['queries'] for v in q['targets'].values()}
        events = copy.deepcopy(g['history_events'])
        for role, es in events.items():
            rule = g['rules'][role]
            for index in range(3, 12):
                for _ in range(200000):
                    d = rng.choice([x for x in range(4, 41) if x % 2 == index % 2])
                    a, c = rng.randrange(1, d), rng.randrange(1, d)
                    text = rng.choice((f'Multiply: {a}/{d} × {c}/{d}',
                        f'What is {a}/{d} times {c}/{d}?',
                        f'Calculate ({a}/{d}) * ({c}/{d}).'))
                    q = dict(problem_text=text, operands=[a,d,c,d], operation='fraction_multiplication')
                    if signatures(q) & seen:
                        rejections['seen_numbers'] += 1
                        continue
                    if target_strings & set(numeric_tokens(history([q], rule))):
                        rejections['target_in_added_history'] += 1
                        continue
                    assert all(answer(r,q) == reference(r,q) for r in RULES)
                    seen.update(signatures(q))
                    es.append(q)
                    break
                else:
                    raise RuntimeError('History sampling exhausted')
        extended[g['pair_id']] = events
    groups = []
    for cohort in COHORTS:
        for source in old['groups']:
            g = copy.deepcopy(source)
            events = extended[g['pair_id']]
            if g['id'].endswith(':1'):
                events = {role:events[other] for role,other in
                    dict(donor='recipient',recipient='donor',same_donor='random',random='same_donor').items()}
            n = {'h3':3, 'h6':6, 'h12':12}.get(cohort)
            g['history_events'] = {role:(es[:n] if n else es[:3]*4) for role,es in events.items()}
            g['histories'] = {role:diagnostic_history(es,g['rules'][role]) for role,es in g['history_events'].items()}
            g['original_group_id'] = g['id']
            g['id'] = cohort+':'+g['id']
            g['cohort'] = cohort
            g['history_length'] = len(g['history_events']['donor'])
            g['n_unique_history_examples'] = n or 3
            g['history_identifiability'] = {role:[r for r in RULES if all(
                answer(r,q)==answer(g['rules'][role],q) for q in es)]
                for role,es in g['history_events'].items()}
            g['prompt_hashes'] = {f'{q["query_idx"]}:{c}':digest(prompt(g,q,c))
                                  for q in g['queries'] for c in CONDITIONS}
            groups.append(g)
    m = dict(protocol='c4_history_diagnostic_v1',seed=SEED,checkpoints=old['checkpoints'],
        rules=old['rules'],groups=groups,conditions=list(CONDITIONS),cohorts=list(COHORTS),
        original_manifest_sha256=sha(root/SOURCE),source_lineage=lineage,
        max_length=768,max_new_tokens=48,n_unordered_pairs=16,
        n_queries_per_cohort=96,n_generations_per_shard=672,
        n_generations_per_checkpoint=2688,rejections=dict(rejections),
        scope='Exploratory post-hoc diagnosis on the prior fixed questions; not new independent replication. No weight updates or rule/checkpoint selection.',
        primary='Per category: paired h12 minus h3 and h12 minus repeat3x4 on open_history, donor_native, and swap-minus-random. h6 is dose-response diagnostic. Keep known controls separate.',
        selection='All four original pairs/category, both directions and all three queries. All outputs retained.',
        bootstrap=dict(unit='original unordered history pair; retain directions, queries, cohorts and arms together',draws=10000,seed=2026090604))
    validate(m, old)
    return m


def validate(m, old=None):
    assert m['protocol']=='c4_history_diagnostic_v1'
    assert m['conditions']==list(CONDITIONS) and m['cohorts']==list(COHORTS)
    assert len(m['groups'])==128 and m['n_generations_per_shard']==672
    assert len({g['id'] for g in m['groups']})==128
    index={(g['cohort'],g['original_group_id']):g for g in m['groups']}
    originals={g['id']:g for g in old['groups']} if old else {}
    for g in m['groups']:
        assert len(g['queries'])==3 and len(set(g['histories'].values()))==4
        h3=index['h3',g['original_group_id']]
        assert g['queries']==h3['queries'] and g['rules']==h3['rules']
        if old:
            assert h3['histories']==originals[g['original_group_id']]['histories']
            assert g['queries']==originals[g['original_group_id']]['queries']
        forbidden={v for q in g['queries'] for v in q['targets'].values()}
        assert not forbidden & set(numeric_tokens(' '.join(g['histories'].values()))), g['id']
        for role,es in g['history_events'].items():
            assert len(es)==g['history_length']
            assert es[:3]==h3['history_events'][role]
            if g['cohort']=='repeat3x4':
                assert es==h3['history_events'][role]*4
            else:
                assert es==index['h12',g['original_group_id']]['history_events'][role][:len(es)]
                assert len({tuple(q['operands']) for q in es})==len(es)
            assert g['histories'][role]==diagnostic_history(es,g['rules'][role])
            assert g['history_identifiability'][role]==[g['rules'][role]]
            for q in es:
                assert answer(g['rules'][role],q)==reference(g['rules'][role],q)
        for q in g['queries']:
            assert q['correct']==correct(q)
            assert all(v==answer(g['rules'][r],q)==reference(g['rules'][r],q) for r,v in q['targets'].items())
            for c in CONDITIONS:
                assert digest(prompt(g,q,c))==g['prompt_hashes'][f'{q["query_idx"]}:{c}']


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/MANIFEST)
    a=p.parse_args();m=build();a.output.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:m[k] for k in ('n_unordered_pairs','n_generations_per_shard','rejections')}))
