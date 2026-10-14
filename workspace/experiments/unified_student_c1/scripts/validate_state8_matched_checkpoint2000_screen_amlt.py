#!/usr/bin/env python3
"""Fail-closed validation for the dynamically hash-bound 7-job screen."""

from __future__ import annotations
import argparse, re
from pathlib import Path
import hashlib
import yaml
from amlt_validation_utils import command_text, target_matches, validate_shell_syntax

IMAGE="amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE="CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
ROOT="/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/checkpoint-2000"
LAUNCHERS={
 'state8-matched-checkpoint2000-t3':'scripts/run_state8_matched_checkpoint_t3.sh',
 'state8-matched-checkpoint2000-answer-c4':'scripts/run_state8_matched_checkpoint_answer_c4.sh',
 **{f'state8-matched-checkpoint2000-diagnosis-c4-shard{s:02d}':'scripts/run_state8_matched_checkpoint_diagnosis_c4_shard.sh' for s in (0,10,20,30,40)},
}

def validate(
    path: Path,
    allow_template: bool = False,
    expected_jobs: set[str] | None = None,
) -> None:
    config=yaml.safe_load(path.read_text()); rendered=yaml.safe_dump(config,sort_keys=False)
    placeholders={"HASH_ADAPTER_SHA256","HASH_STATE_SHA256","HASH_RECEIPT_SHA256"}
    if any(x in rendered for x in placeholders) and not allow_template: raise ValueError("Unresolved hash placeholder")
    if re.search(r"<[^>]+>|\bREQUIRED\b",rendered): raise ValueError("Unresolved placeholder")
    if not target_matches(config.get('target'),WORKSPACE): raise ValueError("Target drift")
    if (config.get('environment') or {}).get('image')!=IMAGE: raise ValueError("Image drift")
    code=config.get('code') or {}
    if code.get('local_dir') is not None:
        if code.get('local_dir')!='$CONFIG_DIR/..': raise ValueError("Code root drift")
        if path.parent.name!='amlt': raise ValueError("Dynamic config must remain in project amlt directory")
    elif code.get('storage_id')!='default':
        raise ValueError("Rendered code storage drift")
    jobs=config.get('jobs') or []; names={str(x.get('name')) for x in jobs}
    expected={'state8-matched-checkpoint2000-t3','state8-matched-checkpoint2000-answer-c4'}|{f'state8-matched-checkpoint2000-diagnosis-c4-shard{s:02d}' for s in (0,10,20,30,40)}
    selected = expected if expected_jobs is None else expected_jobs
    if not selected or not selected <= expected:
        raise ValueError("Invalid expected job subset")
    if names!=selected: raise ValueError("Job set drift")
    starts=[]
    for job in jobs:
        name=job['name']; env=(job.get('submit_args') or {}).get('env') or {}
        if (job.get('sku'),job.get('process_count_per_node'),job.get('mpi'),job.get('sla_tier')) != ('80G1-A100',1,False,'Standard'): raise ValueError(f"Topology drift {name}")
        if env.get('_AZUREML_SINGULARITY_JOB_UAI')!='rankfun_aml' or env.get('STATE8_MATCHED_CHECKPOINT_ROOT')!=ROOT or env.get('STATE8_MATCHED_CHECKPOINT_STEP')!='2000': raise ValueError(f"Environment drift {name}")
        for key in ('STATE8_MATCHED_ADAPTER_SHA256','STATE8_MATCHED_STATE_SHA256','MATCHED_CHECKPOINT_RECEIPT_SHA256'):
            value=str(env.get(key,''))
            if value in placeholders and allow_template: continue
            if len(value)!=64 or any(c not in '0123456789abcdef' for c in value): raise ValueError(f"Digest drift {name}/{key}")
        cmd=command_text(job.get('command')); validate_shell_syntax(cmd,name)
        launcher=Path(__file__).resolve().parents[1]/LAUNCHERS[name]
        digest=hashlib.sha256(launcher.read_bytes()).hexdigest()
        if digest not in cmd: raise ValueError(f"Launcher SHA256 drift {name}")
        if 'diagnosis-c4-shard' in name:
            start=int(env.get('PAIR_START',-1)); starts.append(start)
            if int(env.get('PAIR_COUNT',-1))!=10: raise ValueError("Pair count drift")
            if 'run_state8_matched_checkpoint_diagnosis_c4_shard.sh' not in cmd: raise ValueError("Diagnosis entrypoint drift")
        elif name.endswith('-t3') and 'run_state8_matched_checkpoint_t3.sh' not in cmd: raise ValueError("T3 entrypoint drift")
        elif name.endswith('-answer-c4') and 'run_state8_matched_checkpoint_answer_c4.sh' not in cmd: raise ValueError("Answer entrypoint drift")
    expected_starts = sorted(
        int(name.rsplit("shard", 1)[1])
        for name in selected
        if "diagnosis-c4-shard" in name
    )
    if sorted(starts)!=expected_starts: raise ValueError("Pair coverage drift")

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--allow-template',action='store_true'); p.add_argument('--job',action='append',default=[]); a=p.parse_args(); validate(a.config,a.allow_template,set(a.job) if a.job else None); print(f'validated={a.config}')
if __name__=='__main__': main()
