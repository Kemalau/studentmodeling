"""Derive two inference jobs from the project's accepted Feeds-LLM config."""
import copy
from pathlib import Path
import yaml
from c4_freegen_protocol import sha

ROOT=Path(__file__).resolve().parents[1]
FILES=['requirements-feeds.txt','scripts/copy_model_parallel.py','scripts/train_c1_lora.py',
       'scripts/state_swap_misconception_smoke.py','scripts/c4_freegen_protocol.py',
       'scripts/c4_rule_followup.py','scripts/c4_execution_data.py','scripts/c4_execution_model.py',
       'scripts/c4_unseen_rules.py','scripts/evaluate_c4_unseen_rules.py',
       'scripts/preflight_c4_unseen_rules.py','scripts/run_c4_unseen_rules_feeds.sh',
       'manifests/c4_unseen_rules_v1_20260906.json','reports/c4_unseen_rules_protocol_20260906.md']
HASH='manifests/c4_unseen_rules_v1_feedsllm_20260906.sha256'
CONFIG=ROOT/'amlt/feedsllm_c4_unseen_rules_v1_20260906.yaml'


def build():
    c=yaml.safe_load((ROOT/'amlt/feeds_c4_execution_training_v1_recovery1_20260906.yaml').read_text())
    assert c['target']['name']=='Feeds-LLM' and c['target']['service']=='manifold'
    assert c['target']['workspace_name']=='CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity'
    assert c['environment']['image']=='amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258'
    c['description']='Frozen C4 A/B final checkpoints: unseen-rule generalization; 864 candidate-free generations per arm; inference only'
    c['jobs']=copy.deepcopy(c['jobs'])
    assert len(c['jobs'])==2
    for arm,j in zip(('A','B'),c['jobs']):
        assert j['sku']=='80G1-A100' and j['sla_tier']=='Standard'
        assert j['submit_args']['env']['_AZUREML_SINGULARITY_JOB_UAI']=='rankfun_aml'
        j['name']=f'c4-unseen-rules-arm-{arm.lower()}-seed42'
        j['command']=f'bash scripts/run_c4_unseen_rules_feeds.sh {arm}'
        j['submit_args']['max_run_duration_seconds']=14400
    ignore=['*','!requirements-feeds.txt']
    for folder in ('scripts','manifests','reports'):
        ignore+=['!'+folder+'/',folder+'/*']+['!'+f for f in FILES+[HASH] if f.startswith(folder+'/')]
    c['code']['ignore']=ignore+['__pycache__/','*.pyc']
    return c


if __name__=='__main__':
    c=build()
    CONFIG.write_text('# Derived from accepted feeds_c4_execution_training_v1_recovery1_20260906.yaml.\n'+yaml.safe_dump(c,sort_keys=False))
    (ROOT/HASH).write_text(''.join(sha(ROOT/f)+'  '+f+'\n' for f in FILES))
    print(CONFIG)
    print('Frozen',len(FILES),'files;',sum((ROOT/f).stat().st_size for f in FILES),'bytes')
