"""Validate the frozen panel without regenerating data or reading old datasets."""
import argparse
import json
from pathlib import Path
from c4_unseen_rules import CONDITIONS, RULES, answer, reference, correct, digest, prompt
from c4_freegen_protocol import sha


def validate(m):
    assert m['protocol']=='c4_unseen_rules_v1'
    assert m['conditions']==list(CONDITIONS)
    assert len(m['groups'])==32 and m['n_queries']==96
    assert m['n_generations_per_checkpoint']==864
    assert len({g['id'] for g in m['groups']})==32
    assert len({g['pair_id'] for g in m['groups']})==16
    for g in m['groups']:
        assert len(g['queries'])==3 and len(set(g['histories'].values()))==4
        assert {q['operands'][1]%2 for q in g['queries']}=={0,1}
        for role,events in g['history_events'].items():
            rule=g['rules'][role]
            compatible=[r for r in RULES if all(answer(r,q)==answer(rule,q) for q in events)]
            assert compatible==[rule]
        for q in g['queries']:
            assert q['correct']==correct(q)
            for role,rule in g['rules'].items():
                assert q['targets'][role]==answer(rule,q)==reference(rule,q)
            assert q['targets']['donor']!=q['targets']['recipient']
            for c in CONDITIONS:
                assert digest(prompt(g,q,c))==g['prompt_hashes'][f'{q["query_idx"]}:{c}']


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--arm',choices=('A','B'),required=True)
    p.add_argument('--adapter',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.manifest.read_text());validate(m)
    assert sha(a.adapter/'adapter_model.safetensors')==m['checkpoints'][a.arm]['adapter_sha256']
    config=json.loads((a.adapter/'adapter_config.json').read_text())
    assert config['r']==8
    print('PASS frozen panel, prompt hashes, formulas, and final adapter hash',flush=True)
