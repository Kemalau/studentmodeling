#!/usr/bin/env python3
"""Fail-closed validator for the 20-job same-question final evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE="amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE="CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
CANONICAL="/mnt/amulet/studentmodeling/checkpoints/state8_eedi_same_question_unified_canonical_v1_20260811"
PROGRESS="state8_eedi_same_question_final_full259845"
FORMAL={f"state8-final-formal-shard{i:02d}":i for i in range(16)}
C4={
 "state8-final-c4-seed13":(13,"9856e643848d11a834761e3d28fb078f6eb70fa6c42fc468c3eb1b3a1aafd693"),
 "state8-final-c4-seed42":(42,"edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8"),
 "state8-final-c4-seed97":(97,"15ed710992abd71c37bdfa2246a5a38b5eb422e70ecbfb6da933be013d6d9118"),
}
EEDI="state8-final-eedi-same-question-c4-final150"


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
    rendered=yaml.safe_dump(config,sort_keys=False)
    if re.search(r"<placeholder>|\bREQUIRED\b",rendered): raise ValueError("placeholder")
    if not target_matches(config.get("target"),WORKSPACE): raise ValueError("target")
    if (config.get("environment") or {}).get("image")!=IMAGE: raise ValueError("image")
    if (config.get("storage") or {}).get("student_store")!={
      "storage_account_name":"csnewsandfeeds4150361735","container_name":"amulet",
      "mount_dir":"/mnt/amulet","is_output":True}: raise ValueError("storage")
    jobs=config.get("jobs") or []; full=set(FORMAL)|set(C4)|{EEDI}
    expected=full if expected_jobs is None else expected_jobs
    names={str(job.get("name")) for job in jobs}
    if not expected or not expected.issubset(full) or names!=expected:
        raise ValueError(f"job set drift {sorted(names)}")
    for job in jobs:
        name=str(job["name"]); submit=job.get("submit_args") or {}; env=submit.get("env") or {}
        if job.get("sku")!="80G1-A100" or int(job.get("process_count_per_node",0))!=1 \
           or job.get("mpi") is not False or job.get("sla_tier")!="Standard":
            raise ValueError(f"topology {name}")
        if submit.get("max_attempts")!=2 or submit.get("max_run_duration_seconds")!=259200:
            raise ValueError(f"duration {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI")!="rankfun_aml": raise ValueError(f"uai {name}")
        if env.get("VENV_ROOT")!="/tmp/studentmodeling-state8-final-eval-venv": raise ValueError(f"venv {name}")
        if env.get("STATE8_FINAL_EVAL_RUN_DIR")!=CANONICAL: raise ValueError(f"canonical {name}")
        if env.get("STATE8_FINAL_EVAL_PROGRESS_NAMESPACE")!=PROGRESS: raise ValueError(f"progress {name}")
        command=command_text(job.get("command")); validate_shell_syntax(command,name)
        if name in FORMAL:
            if str(env.get("NUM_SHARDS"))!="16" or str(env.get("SHARD_INDEX"))!=str(FORMAL[name]):
                raise ValueError(f"shard {name}")
            if "run_state8_eedi_same_question_final_formal_shard.sh" not in command:
                raise ValueError(f"formal runner {name}")
        elif name in C4:
            seed,digest=C4[name]
            if str(env.get("PAIR_SEED"))!=str(seed) or env.get("PAIR_MANIFEST_SHA256")!=digest:
                raise ValueError(f"C4 coordinate {name}")
            if "run_state8_eedi_same_question_final_malrule_c4.sh" not in command:
                raise ValueError(f"C4 runner {name}")
        else:
            if set(env)!={"_AZUREML_SINGULARITY_JOB_UAI","VENV_ROOT","STATE8_FINAL_EVAL_RUN_DIR","STATE8_FINAL_EVAL_PROGRESS_NAMESPACE"}:
                raise ValueError("Eedi env")
            if "run_state8_eedi_same_question_final_eedi_c4.sh" not in command:
                raise ValueError("Eedi runner")


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--jobs",nargs="*")
    a=p.parse_args(); value=yaml.safe_load(a.config.read_text())
    validate(value,set(a.jobs) if a.jobs else None); print("valid=true")


if __name__=="__main__": main()
