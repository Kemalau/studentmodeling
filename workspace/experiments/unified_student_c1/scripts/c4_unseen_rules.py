"""Outcome-free unseen-rule panel for the two frozen execution checkpoints.

No model inference or optimization. Novelty means absent as a labelled rule in
the audited fine-tuning sources, never absence from pretraining. Conditional
rules deliberately compose seen components; their novelty is cross-query routing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from math import gcd
from pathlib import Path
import random

from c4_execution_data import INSTRUCTION, STATE
from c4_freegen_protocol import numeric_tokens, signatures, sha

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026090603
RULES = {
    'offset_3': 'Ignore both denominators. Multiply the two written numerators, then add exactly 3. Return the resulting integer.',
    'offset_7': 'Ignore both denominators. Multiply the two written numerators, then add exactly 7. Return the resulting integer.',
    'shift_1_keep': 'Add exactly 1 to each written numerator, multiply those two shifted numerators, and divide by the common written denominator once. Simplify.',
    'shift_2_keep': 'Add exactly 2 to each written numerator, multiply those two shifted numerators, and divide by the common written denominator once. Simplify.',
    'even_ignore_odd_keep': 'If the common written denominator is even, multiply the written numerators and ignore the denominators. If it is odd, multiply the written numerators and divide by the common denominator once. Simplify.',
    'even_keep_odd_ignore': 'If the common written denominator is even, multiply the written numerators and divide by the common denominator once. If it is odd, multiply the written numerators and ignore the denominators. Simplify.',
    'seen_ignore': 'Multiply the two written numerators and ignore both denominators. Return the resulting integer.',
    'seen_keep': 'Multiply the two written numerators and divide by the common written denominator once. Simplify.',
}
PAIRS = {
    'offset': ('offset_3', 'offset_7'),
    'composition': ('shift_1_keep', 'shift_2_keep'),
    'conditional': ('even_ignore_odd_keep', 'even_keep_odd_ignore'),
    'seen_control': ('seen_ignore', 'seen_keep'),
}
CONDITIONS = ('native', 'donor_native', 'noop', 'swap', 'random',
              'same_donor', 'oracle', 'open_history', 'empty_history')
OLD_MANIFEST = 'manifests/c4_execution_training_v1_20260906.json'
CHECKPOINTS = {
    'A': {
        'adapter_sha256': '7d614cf19d847aa220d7b87c8d471ada95867476653425188d633747efccd1b6',
        'path': '/mnt/amulet/projects/v-yuejiali/amlt-results/7211303140.00646-9f11fa46-2ae7/execution_A/final_adapter',
    },
    'B': {
        'adapter_sha256': '87a9bee0d0bde5be481ed95f2209d91807c974a190953aae3b1fa33a0eb24a32',
        'path': '/mnt/amulet/projects/v-yuejiali/amlt-results/7211303140.00542-f9ce2af5-f3be/execution_B/final_adapter',
    },
}


def answer(rule, q):
    a, d, c, e = q['operands']
    assert d == e and d > 0
    if rule.startswith('offset_'):
        return str(a*c + int(rule.split('_')[1]))
    if rule.startswith('shift_'):
        k = int(rule.split('_')[1])
        return str(Fraction((a+k)*(c+k), d))
    ignore = rule == 'seen_ignore'
    if rule == 'even_ignore_odd_keep':
        ignore = d % 2 == 0
    elif rule == 'even_keep_odd_ignore':
        ignore = d % 2 == 1
    elif rule not in ('seen_ignore', 'seen_keep'):
        raise ValueError(rule)
    return str(Fraction(a*c, 1 if ignore else d))


def reference(rule, q):
    """Separate integer-pair implementation, with explicit rational reduction."""
    x, den, y, den2 = q['operands']
    assert den == den2
    formulas = {
        'offset_3': (x*y+3, 1), 'offset_7': (x*y+7, 1),
        'shift_1_keep': (x*y+x+y+1, den),
        'shift_2_keep': (x*y+2*x+2*y+4, den),
        'even_ignore_odd_keep': (x*y, 1 if not den % 2 else den),
        'even_keep_odd_ignore': (x*y, den if not den % 2 else 1),
        'seen_ignore': (x*y, 1), 'seen_keep': (x*y, den),
    }
    n, d = formulas[rule]
    g = gcd(n, d); n //= g; d //= g
    return str(n) if d == 1 else f'{n}/{d}'


def correct(q):
    a, b, c, d = q['operands']
    return str(Fraction(a*c, b*d))


def history(events, rule):
    lines = ['<student_history>']
    for i, q in enumerate(events, 1):
        lines.extend([f'Trial {i}:', 'Problem: '+q['problem_text'],
                      'Student answer: '+answer(rule, q),
                      'Correct answer: '+correct(q), 'Outcome: incorrect', ''])
    return '\n'.join(lines).rstrip()+'\n</student_history>\n'


def prompt(group, query, condition):
    assert condition in CONDITIONS
    role = 'donor' if condition in ('donor_native', 'open_history') else 'recipient'
    context = group['histories'][role]
    if condition in ('oracle', 'empty_history'):
        context = '<student_history>\n</student_history>\n'
    text = INSTRUCTION+context+STATE+'\n<query>\n'
    if condition == 'oracle':
        text += 'Student rule: Follow this rule even if mathematically incorrect: '+RULES[group['rules']['donor']]+'\n'
    return text+'Problem: '+query['problem_text']+'\n</query>\nAnswer:'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def old_sources(root):
    """Audit labelled rules and exclude old number combinations, not just IDs."""
    files = [root/'data/processed/state_curriculum_stage_a_v1'/name
             for name in ('train.jsonl', 'valid.jsonl')]
    files += [root/'data/processed/malrule_c1_clean'/name
              for name in ('train.jsonl', 'val.jsonl', 'test_same_template.jsonl', 'test_cross_template.jsonl')]
    seen, lineage = set(), []
    for path in files:
        labels = Counter(); count = 0
        for line in path.open():
            row = json.loads(line); count += 1
            labels[row.get('metadata', {}).get('malrule_id', 'missing')] += 1
            for q in row.get('history', [])+[row['query']]:
                try:
                    seen.update(signatures(q))
                except (ValueError, ZeroDivisionError):
                    seen.add(('text', tuple(sorted(numeric_tokens(q['problem_text'])))))
        lineage.append(dict(path=str(path.relative_to(root)), sha256=sha(path),
                            rows=count, labelled_rules=dict(labels)))
    previous = json.loads((root/OLD_MANIFEST).read_text())
    for groups in previous['splits'].values():
        for g in groups:
            for events in g['history_events'].values():
                for q in events: seen.update(signatures(q))
            for q in g['queries']: seen.update(signatures(q))
    lineage.append(dict(path=OLD_MANIFEST, sha256=sha(root/OLD_MANIFEST),
                        labelled_rules=previous['rules']))
    other = root/'manifests/c4_freegen_novel_seed42_20260905.json'
    for g in json.loads(other.read_text())['groups']:
        for q in g['queries']: seen.update(signatures(q))
    lineage.append(dict(path=str(other.relative_to(root)), sha256=sha(other)))
    return seen, lineage


def build(root=ROOT, pairs_per_category=4):
    assert pairs_per_category > 0
    seen, lineage = old_sources(root)
    rng = random.Random(SEED); rejected = Counter(); checks = 0
    def sample(pool, parity, forbidden=(), prior=()):
        nonlocal checks
        for _ in range(200000):
            d = rng.choice([v for v in range(4, 41) if v % 2 == parity])
            a = rng.randrange(1, d); c = rng.randrange(1, d)
            text = rng.choice((f'Multiply: {a}/{d} × {c}/{d}',
                               f'What is {a}/{d} times {c}/{d}?',
                               f'Calculate ({a}/{d}) * ({c}/{d}).'))
            q = dict(problem_text=text, operands=[a,d,c,d], operation='fraction_multiplication')
            sig = signatures(q)
            if sig & seen: rejected['seen_numbers'] += 1; continue
            ys = [answer(r,q) for r in pool]
            if len(set(ys+[correct(q)])) != 3:
                rejected['answer_collision'] += 1; continue
            if any(y in forbidden or y in prior for y in ys):
                rejected['copy_or_repeated_target'] += 1; continue
            for rule in RULES:
                assert answer(rule,q) == reference(rule,q)
                checks += 1
            seen.update(sig)
            return q
        raise RuntimeError(f'Sampling exhausted: {pool}, {dict(rejected)}')

    groups = []
    for category, pool in PAIRS.items():
        for pi in range(pairs_per_category):
            # Both parity branches appear in every history and query triplet.
            parities = [pi%2, 1-pi%2, pi%2]
            rules = dict(a=pool[0], b=pool[1], same_a=pool[0], same_b=pool[1])
            events = {}; histories = {}
            for role, rule in rules.items():
                for _ in range(1000):
                    es = [sample(pool,p) for p in parities]
                    compatible = [r for r in RULES if all(answer(r,q)==answer(rule,q) for q in es)]
                    if compatible == [rule]: break
                    rejected['ambiguous_history_within_declared_8_rules'] += 1
                else: raise RuntimeError('No identifiable history')
                events[role] = es; histories[role] = history(es,rule)
            forbidden = set(numeric_tokens(' '.join(histories.values())))
            queries = []; prior = set()
            for qi, parity in enumerate(parities):
                q = sample(pool,parity,forbidden,prior)
                prior.update(answer(r,q) for r in pool)
                queries.append(dict(q,query_idx=qi))
            for reverse in (False,True):
                donor, recipient = ('b','a') if reverse else ('a','b')
                mapping = dict(donor=donor,recipient=recipient,
                               same_donor='same_'+donor,random='same_'+recipient)
                g = dict(id=f'{category}:{pi:03d}:{int(reverse)}',
                         pair_id=f'{category}:{pi:03d}',category=category,
                         rules={k:rules[v] for k,v in mapping.items()},
                         histories={k:histories[v] for k,v in mapping.items()},
                         history_events={k:events[v] for k,v in mapping.items()},queries=[])
                for q in queries:
                    g['queries'].append(dict(q,correct=correct(q),
                        targets={k:answer(r,q) for k,r in g['rules'].items()}))
                g['history_identifiability'] = {
                    role:[r for r in RULES if all(answer(r,q)==answer(g['rules'][role],q) for q in es)]
                    for role,es in g['history_events'].items()}
                # Finite rule-set evidence only; arbitrarily many functions can
                # fit any finite history. Do not claim absolute identifiability.
                g['query_signature_compatible_rules'] = [r for r in RULES
                    if all(answer(r,q)==q['targets']['donor'] for q in g['queries'])]
                assert g['query_signature_compatible_rules'] == [g['rules']['donor']]
                g['prompt_hashes'] = {f'{q["query_idx"]}:{c}':digest(prompt(g,q,c))
                                      for q in g['queries'] for c in CONDITIONS}
                groups.append(g)
    manifest = dict(protocol='c4_unseen_rules_v1', seed=SEED, rules=RULES,
        pairs=PAIRS, checkpoints=CHECKPOINTS, groups=groups, source_lineage=lineage,
        rejections=dict(rejected), independent_answer_checks=checks,
        conditions=CONDITIONS, max_new_tokens=48, max_length=768,
        n_unordered_pairs=len(PAIRS)*pairs_per_category,
        n_queries=len(groups)*3, n_generations_per_checkpoint=len(groups)*3*len(CONDITIONS),
        scope='Inference-only exploratory test of constructed rules absent as labelled rules from audited fine-tuning sources. Not unseen pretraining, invented rules, arbitrary-program execution, or human fidelity. Conditional rules compose seen outputs.',
        primary='Per-category swap-minus-wrong-rule donor-target accuracy; paired B-minus-A. Keep seen-control separate. Joint-three is essential for conditional rules.',
        bootstrap=dict(unit='unordered history pair, both directions and all queries',draws=10000,seed=2026090604),
        selection='Four preselected categories, four pairs/category, all outcomes retained; no test-based rule/checkpoint/example selection.')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); m = build()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:m[k] for k in ('n_unordered_pairs','n_queries','n_generations_per_checkpoint','independent_answer_checks','rejections')},indent=2))
