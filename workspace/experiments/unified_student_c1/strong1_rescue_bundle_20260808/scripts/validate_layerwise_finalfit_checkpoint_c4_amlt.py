#!/usr/bin/env python3
"""Fail-closed AMLT validator for the formal layerwise checkpoint C4 probe."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
DEFAULT_ROUND = 1000
DEFAULT_RUN_DIR = "/mnt/amulet/studentmodeling/diagnostics/layerwise_unified_finalfit_seed42_20260801_snapshot"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [index for marker in ("description:", "target:") if (index := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(
    config: dict,
    checkpoint_round: int = DEFAULT_ROUND,
    *,
    job_name: str | None = None,
    run_dir: str = DEFAULT_RUN_DIR,
) -> None:
    if checkpoint_round <= 0:
        raise ValueError("Checkpoint round must be positive")
    job_name = job_name or f"layerwise-finalfit-checkpoint{checkpoint_round}-c4-seed42"
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    expected_storage = {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }
    if storage != expected_storage:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != job_name:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    if (
        submit.get("max_attempts") != 2
        or submit.get("max_run_duration_seconds") != 172800
        or (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
    ):
        raise ValueError("Retry/duration/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, job_name)
    required = (
        f"export RUN_DIR={run_dir}",
        f"checkpoint-round-{checkpoint_round}",
        "adapter/adapter_config.json",
        f"receipt.get('round') == {checkpoint_round}",
        "receipt.get('world_size') == 16",
        "checkpoint_completion.json",
        "receipt['sha256']['adapter_model']",
        "receipt['sha256']['recurrent_state']",
        "audit_layerwise_distributed_checkpoint.py",
        "checkpoint_resume_audit.json",
        f"--expected-round {checkpoint_round}",
        "6690ece8935c6343bec7136ad34a9899b72a866c7bde6286ca26b2c82890007f",
        "--state-architecture layerwise",
        "--num-pairs 150",
        "c4_stage_a_gate_pairs_seed42_n150_v1.json",
        "<|vision_pad|>",
        "diagnostic_only=true",
        "formal_final_c4_required=true",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "fd6510465abafbd897ac4acbf27dd0da14b179637ba9a632f4c5cb539214c5ad",
        "c4_eval_progress_v1",
        "--progress-dir",
        "--progress-key",
        "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
        "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
        "scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py",
        "tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py",
        "cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py",
        "cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py",
    )
    if any(value not in command for value in required):
        raise ValueError("Checkpoint C4 command drift")
    forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models", "--state-architecture scalar")
    if any(value.lower() in command.lower() for value in forbidden):
        raise ValueError("Secret/local/scalar path leaked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--round", type=int, default=DEFAULT_ROUND)
    parser.add_argument("--job-name")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    validate(
        load(args.config),
        args.round,
        job_name=args.job_name,
        run_dir=args.run_dir,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
