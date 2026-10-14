#!/usr/bin/env python3
"""Fail-closed preflight for the matched checkpoint-3000 rescue."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
ROOT = "/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/checkpoint-3000"
JOB = "state8-matched-cp3000-rescue-semantic-preserve"
HASHES = {
    "adapter": "3f2d86dff15cc892232f4810fbba05d854660ab42513d76d08cb6831a6767f8f",
    "state": "e61eccba80c9d28b083c5b6374f2bee1369ccb25dcccd8e3607c8ef9df9714c5",
}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text())
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b|HASH_RUNNER", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage=(config.get("storage") or {}).get("student_store")
    if storage != {"storage_account_name":"csnewsandfeeds4150361735","container_name":"amulet","mount_dir":"/mnt/amulet","is_output":True}:
        raise ValueError("Storage drift")
    code=config.get("code") or {}
    if not (code.get("local_dir")=="$CONFIG_DIR/.." or code.get("storage_id")=="default"):
        raise ValueError("Code root drift")
    jobs=config.get("jobs") or []
    if len(jobs)!=1 or jobs[0].get("name")!=JOB:
        raise ValueError("Job drift")
    job=jobs[0]; submit=job.get("submit_args") or {}; env=submit.get("env") or {}
    if (job.get("sku"),job.get("process_count_per_node"),job.get("mpi"),job.get("sla_tier")) != ("80G1-A100",1,False,"Standard"):
        raise ValueError("Topology drift")
    if submit.get("max_attempts")!=2 or submit.get("max_run_duration_seconds")!=86400:
        raise ValueError("Retry/duration drift")
    expected={
      "_AZUREML_SINGULARITY_JOB_UAI":"rankfun_aml",
      "VENV_ROOT":"/tmp/studentmodeling-state8-matched-cp3000-rescue-venv",
      "MATCHED_CHECKPOINT_ROOT":ROOT,
      "EXPECTED_STUDENT_ADAPTER_SHA256":HASHES["adapter"],
      "EXPECTED_STUDENT_STATE_SHA256":HASHES["state"],
      "STATE8_RESCUE_MODE":"semantic_preserve",
      "RUN_TAG":"matched_cp3000_semantic_preserve_v1",
    }
    if env!=expected:
        raise ValueError("Environment/artifact binding drift")
    command=command_text(job.get("command")); validate_shell_syntax(command,JOB)
    if "run_state8_matched_checkpoint3000_semantic_rescue.sh" not in command:
        raise ValueError("Entrypoint drift")
    scripts=Path(__file__).resolve().parent
    runner=scripts/"run_state8_matched_checkpoint3000_semantic_rescue.sh"
    digest=hashlib.sha256(runner.read_bytes()).hexdigest()
    if digest not in command:
        raise ValueError("Runner hash drift")
    base=(scripts/"run_state8_matched_checkpoint2000_rescue.sh")
    if hashlib.sha256(base.read_bytes()).hexdigest()!="c04d4059c8f94a4cb3b1146f2ef470c8943acf5b44e8e665c422265e45bd29ab":
        raise ValueError("Frozen cp2000 launcher drift")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args(); validate(args.config); print(f"validated={args.config}")


if __name__=="__main__":
    main()
