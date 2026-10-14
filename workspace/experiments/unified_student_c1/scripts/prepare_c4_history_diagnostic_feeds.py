"""Derive fixed history diagnostic shards from the accepted project config."""
import copy
import json
from pathlib import Path
import yaml
from c4_history_diagnostic import ROOT, COHORTS, MANIFEST, validate
from c4_freegen_protocol import sha

CONFIG=ROOT/'amlt/feedsllm_c4_history_diagnostic_v1_20260907.yaml'
HASH='manifests/c4_history_diagnostic_v1_20260907.sha256'
FILES=['requirements-feeds.txt','scripts/copy_model_parallel.py','scripts/train_c1_lora.py',
    'scripts/state_swap_misconception_smoke.py','scripts/c4_freegen_protocol.py',
    'scripts/c4_rule_followup.py','scripts/c4_execution_data.py','scripts/c4_execution_model.py',
    'scripts/c4_unseen_rules.py','scripts/evaluate_c4_unseen_rules.py',
    'scripts/c4_history_diagnostic.py','scripts/evaluate_c4_history_diagnostic.py',
    'scripts/run_c4_history_diagnostic.sh',MANIFEST,'reports/c4_history_diagnostic_protocol_20260907.md']


def build():
    old=yaml.safe_load((ROOT/'amlt/feedsllm_c4_unseen_rules_v1_20260906.yaml').read_text())
    assert old['target']==dict(service='manifold',name='Feeds-LLM',workspace_name='CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity')
    assert old['environment']['image']=='amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258'
    config=copy.deepcopy(old);config['description']='Frozen A/B C4 history diagnosis: 3/6/12 unique examples and repeated-3 control; 5376 free generations; no training'
    config['jobs']=[]
    for arm in ('A','B'):
        for cohort in COHORTS:
            job=copy.deepcopy(old['jobs'][0])
            assert job['sku']=='80G1-A100' and job['sla_tier']=='Standard'
            assert job['submit_args']['env']['_AZUREML_SINGULARITY_JOB_UAI']=='rankfun_aml'
            job['name']=f'c4-history-{arm.lower()}-{cohort}'
            job['command']=f'bash scripts/run_c4_history_diagnostic.sh {arm} {cohort}'
            config['jobs'].append(job)
    ignore=['*','!requirements-feeds.txt']
    for folder in ('scripts','manifests','reports'):
        ignore+=['!'+folder+'/',folder+'/*']+['!'+f for f in FILES+[HASH] if f.startswith(folder+'/')]
    config['code']['ignore']=ignore+['__pycache__/','*.pyc']
    return config


if __name__=='__main__':
    validate(json.loads((ROOT/MANIFEST).read_text()))
    CONFIG.write_text('# Derived from accepted feedsllm_c4_unseen_rules_v1_20260906.yaml.\n'+yaml.safe_dump(build(),sort_keys=False))
    (ROOT/HASH).write_text(''.join(sha(ROOT/f)+'  '+f+'\n' for f in FILES))
    print(json.dumps(dict(config=str(CONFIG),frozen_files=len(FILES),bytes=sum((ROOT/f).stat().st_size for f in FILES))))
