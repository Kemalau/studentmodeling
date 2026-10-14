"""Derive AMLT config from accepted project config, explicit file allowlist."""
import argparse
import copy
from pathlib import Path
import yaml
from c4_freegen_protocol import sha
ROOT=Path(__file__).resolve().parents[1]
FILES=['requirements-feeds.txt','scripts/copy_model_parallel.py','scripts/train_c1_lora.py',
       'scripts/state_swap_misconception_smoke.py','scripts/c4_freegen_protocol.py','scripts/c4_rule_followup.py',
       'scripts/c4_execution_data.py','scripts/c4_execution_model.py','scripts/run_c4_execution_training.py',
       'scripts/run_c4_execution_training.sh','tests/test_c4_execution_training.py',
       'manifests/c4_execution_training_v1_20260906.json','reports/c4_execution_training_protocol_20260906.md']
HASH='manifests/c4_execution_training_v1_20260906.sha256'

def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=('Feeds','Feeds-LLM'),default='Feeds');a=p.parse_args()
    c=yaml.safe_load((ROOT/'amlt/feeds_c4_rule_followup_v2_seed42_20260905.yaml').read_text())
    assert c['target']['name']=='Feeds';c['target']['name']=a.target
    c['description']='C4 execution repair pilot: matched candidate-free SFT versus differentiable state exchange; 240 steps, six rules, seed42'
    c['jobs']=copy.deepcopy(c['jobs'][:2])
    for arm,j in zip(('A','B'),c['jobs']):
        j['submit_args']=copy.deepcopy(j['submit_args'])
        j['name']=f'c4-execution-arm-{arm.lower()}-seed42';j['command']=f'bash scripts/run_c4_execution_training.sh {arm}'
        j['submit_args']['max_run_duration_seconds']=28800
    ignore=['*','!requirements-feeds.txt']
    for folder in ('scripts','tests','manifests','reports'):
        ignore += ['!'+folder+'/',folder+'/*']+['!'+f for f in FILES+[HASH] if f.startswith(folder+'/')]
    c['code']['ignore']=ignore+['__pycache__/','*.pyc']
    dest=ROOT/'amlt/feeds_c4_execution_training_v1_20260906.yaml'
    dest.write_text('# Derived from accepted feeds_c4_rule_followup_v2_seed42_20260905.yaml.\n'+yaml.safe_dump(c,sort_keys=False))
    (ROOT/HASH).write_text(''.join(sha(ROOT/f)+'  '+f+'\n' for f in FILES))
    print(dest);print('Frozen',len(FILES),'files;',sum((ROOT/f).stat().st_size for f in FILES),'bytes')

if __name__=='__main__':main()
