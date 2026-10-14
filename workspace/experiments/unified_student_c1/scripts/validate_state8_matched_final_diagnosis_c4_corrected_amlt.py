#!/usr/bin/env python3
from __future__ import annotations
import argparse,re
from pathlib import Path
import yaml
from amlt_validation_utils import command_text,target_matches,validate_shell_syntax

IMAGE='amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258'
WORKSPACE='CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity'
JOB='state8-matched-final-diagnosis-c4-corrected'

def validate(path: Path, allow_template: bool=False) -> None:
    x=yaml.safe_load(path.read_text()); rendered=yaml.safe_dump(x,sort_keys=False)
    placeholders={'HASH_ADAPTER_SHA256','HASH_STATE_SHA256','HASH_RECEIPT_SHA256','HASH_LAUNCHER_SHA256'}
    if any(v in rendered for v in placeholders) and not allow_template: raise ValueError('Unresolved hash placeholder')
    if re.search(r'<[^>]+>|\bREQUIRED\b',rendered): raise ValueError('Unresolved placeholder')
    if not target_matches(x.get('target'),WORKSPACE): raise ValueError('Target drift')
    if (x.get('environment') or {}).get('image')!=IMAGE: raise ValueError('Image drift')
    code=x.get('code') or {}
    if code.get('local_dir') is not None:
        if code.get('local_dir')!='$CONFIG_DIR/..' or path.parent.name!='amlt': raise ValueError('Code root drift')
    elif code.get('storage_id')!='default': raise ValueError('Rendered code storage drift')
    if (x.get('storage') or {}).get('student_store')!={'storage_account_name':'csnewsandfeeds4150361735','container_name':'amulet','mount_dir':'/mnt/amulet','is_output':True}: raise ValueError('Storage drift')
    jobs=x.get('jobs') or []
    if len(jobs)!=1 or jobs[0].get('name')!=JOB: raise ValueError('Job drift')
    job=jobs[0]
    if (job.get('sku'),job.get('process_count_per_node'),job.get('mpi'),job.get('sla_tier'))!=('80G1-A100',1,False,'Standard'): raise ValueError('Topology drift')
    submit=job.get('submit_args') or {}; env=submit.get('env') or {}
    if submit.get('max_attempts')!=2 or submit.get('max_run_duration_seconds')!=21600: raise ValueError('Retry/duration drift')
    if env.get('_AZUREML_SINGULARITY_JOB_UAI')!='rankfun_aml': raise ValueError('UAI drift')
    for key in ('EXPECTED_ADAPTER_SHA256','EXPECTED_STATE_SHA256','EXPECTED_RECEIPT_SHA256'):
        value=str(env.get(key,''))
        if value in placeholders and allow_template: continue
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value): raise ValueError(f'Digest drift {key}')
    cmd=command_text(job.get('command')); validate_shell_syntax(cmd,JOB)
    if 'run_state8_matched_final_diagnosis_c4_corrected.sh' not in cmd: raise ValueError('Entrypoint drift')
    launcher_value=next((v for v in cmd.split() if len(v)==64 and all(c in '0123456789abcdef' for c in v)),None)
    if launcher_value is None and not allow_template: raise ValueError('Launcher hash drift')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--allow-template',action='store_true'); a=p.parse_args(); validate(a.config,a.allow_template); print(f'validated={a.config}')
if __name__=='__main__': main()
