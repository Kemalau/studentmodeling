#!/usr/bin/env python3
"""Fail-closed AMLT validator for layerwise final evaluation."""
from __future__ import annotations
import argparse,re
from pathlib import Path
import yaml
from amlt_validation_utils import command_text,target_matches,validate_shell_syntax
IMAGE="amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE="CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
FORMAL_JOBS={"layerwise-final-eval-t1-t6-seed42","layerwise-final-eval-t2-seed42","layerwise-final-eval-t3-seed42","layerwise-final-eval-t5-seed42"}
C4_JOBS={f"layerwise-final-eval-c4-seed{seed}" for seed in (13,42,97)}
JOBS=FORMAL_JOBS|C4_JOBS
C4_MANIFEST_SHA256={
    13:"9856e643848d11a834761e3d28fb078f6eb70fa6c42fc468c3eb1b3a1aafd693",
    42:"edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8",
    97:"15ed710992abd71c37bdfa2246a5a38b5eb422e70ecbfb6da933be013d6d9118",
}
FROZEN_RUNTIME=(
    "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
    "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
    "scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py",
    "tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py",
    "cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py",
    "cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py",
)
def load(path:Path)->dict:
    x=yaml.safe_load(path.read_text());
    if not isinstance(x,dict): raise ValueError("Invalid config")
    return x
def validate(config:dict, expected_jobs:set[str]|None=None)->None:
    text=yaml.safe_dump(config,sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b",text): raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in text: raise ValueError("Forbidden image")
    if not target_matches(config.get('target'),WORKSPACE): raise ValueError("Target drift")
    if (config.get('environment') or {}).get('image')!=IMAGE: raise ValueError("Image drift")
    storage=(config.get('storage') or {}).get('student_store') or {}
    if storage!={"storage_account_name":"csnewsandfeeds4150361735","container_name":"amulet","mount_dir":"/mnt/amulet","is_output":True}: raise ValueError("Storage drift")
    jobs=config.get('jobs') or []
    expected=JOBS if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(JOBS): raise ValueError("Expected job drift")
    if len(jobs)!=len(expected) or {j.get('name') for j in jobs}!=expected: raise ValueError("Job drift")
    commands=[]
    for j in jobs:
        if j.get('sku')!='80G1-A100' or j.get('mpi') is not False or int(j.get('process_count_per_node',0))!=1 or j.get('sla_tier')!='Standard': raise ValueError("GPU/MPI/SLA drift")
        s=j.get('submit_args') or {}; env=s.get('env') or {}
        if s.get('max_attempts')!=2 or env.get('_AZUREML_SINGULARITY_JOB_UAI')!='rankfun_aml': raise ValueError("Retry/UAI drift")
        c=command_text(j.get('command')); validate_shell_syntax(c,str(j.get('name'))); commands.append(c)
        if 'df --output=avail /tmp' not in c or '-ge 83886080' not in c:
            raise ValueError("Ephemeral disk gate drift")
        if any(fragment not in c for fragment in FROZEN_RUNTIME):
            raise ValueError("Final-eval frozen runtime command drift")
    joined='\n'.join(commands)
    required=["layerwise_world_model_v1/final_fit/seed42.json","--state-architecture layerwise","real_examples']==259845","one_epoch_exact","/mnt/amulet/studentmodeling/models/Qwen3-32B"]
    if expected&FORMAL_JOBS:
        required.extend(["7bdb9b7b6e9f4c37db4c8551bba8eb9260a989e707fe535ff6922d03fe082cd2","a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525","raw_history_visible_to_decoder=false","persistent_history_kv_cache=false"])
    if expected&C4_JOBS:
        required.extend([
            "fd6510465abafbd897ac4acbf27dd0da14b179637ba9a632f4c5cb539214c5ad",
            "C4_PAIR_SEED",
            "PAIR_MANIFEST",
            "--progress-dir",
            "--progress-key",
        ])
    if any(x not in joined for x in required): raise ValueError("Final-eval command drift")
    for job in jobs:
        name=str(job.get('name'))
        if name not in C4_JOBS:continue
        seed=int(name.rsplit('seed',1)[1])
        command=command_text(job.get('command'))
        env=((job.get('submit_args') or {}).get('env') or {})
        if str(env.get('C4_PAIR_SEED'))!=str(seed):raise ValueError("C4 pair-seed env drift")
        required_literal=(C4_MANIFEST_SHA256[seed],)
        equivalent_forms=(
            ("c4_stage_a_gate_pairs_seed$${C4_PAIR_SEED}_n150_v1.json", "c4_stage_a_gate_pairs_seed${C4_PAIR_SEED}_n150_v1.json"),
            ("c4_eval_progress_seed$${C4_PAIR_SEED}_v1", "c4_eval_progress_seed${C4_PAIR_SEED}_v1"),
            ('--seed "$$C4_PAIR_SEED"', '--seed "$C4_PAIR_SEED"'),
        )
        if any(fragment not in command for fragment in required_literal) or any(
            not any(form in command for form in forms) for forms in equivalent_forms
        ):
            raise ValueError("C4 pair-seed command drift")
    forbidden=("password","account_key","access_token","/home/v-yuejiali/studentmodeling/models","--state-architecture scalar")
    if any(x.lower() in joined.lower() for x in forbidden): raise ValueError("Secret/local/scalar path leaked")
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--jobs',nargs='*');a=p.parse_args();validate(load(a.config),set(a.jobs) if a.jobs else None);print(f"validated={a.config}")
if __name__=='__main__':main()
