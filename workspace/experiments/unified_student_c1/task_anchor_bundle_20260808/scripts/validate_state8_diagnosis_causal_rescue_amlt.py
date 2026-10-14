#!/usr/bin/env python3
"""Fail-closed Feeds preflight for the State8 diagnosis-causal rescue pilot."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "state8-diagnosis-causal-rescue-seed43"
FILES = {
    "run_state8_diagnosis_causal_rescue.sh": "9280e2c231fb51120abee965c2411b6d4bc626acc495b7f19a3e054adc42ec23",
    "select_latest_valid_distributed_checkpoint.py": "033de45a0d16eb5d36fe9f8552bdebad7e7c8db2cc5d4cdd2a8e1084773aaeb2",
    "train_recurrent_student_m4.py": "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
    "train_recurrent_student_m4_state8_external_teacher_curriculum.py": "8e08385392811e48cfc8cf6335b8e3951641a7a9445e140e37d9d9872d515c6f",
    "augment_recurrent_causal_diagnosis.py": "b74684001e54bf0c9b023540c76a269708f1e8fa529cfec703a4439288590926",
    "evaluate_recurrent_diagnosis_c4.py": "e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store")
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if not (
        str(code.get("local_dir")) == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusions drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job set drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G4-A100"
        or int(job.get("process_count_per_node", 0)) != -1
        or job.get("mpi") is not True
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry/duration drift")
    if (submit.get("env") or {}) != {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-diagnosis-rescue-venv",
    }:
        raise ValueError("Environment/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    for fragment in (
        "set -euo pipefail",
        "FAILED line=%s rc=%s command=%s",
        "bash scripts/run_state8_diagnosis_causal_rescue.sh",
    ):
        if fragment not in command:
            raise ValueError(f"Launcher drift: {fragment}")
    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")
    launcher = (root / "run_state8_diagnosis_causal_rescue.sh").read_text()
    for fragment in (
        "state8_diagnosis_causal_rescue_seed43_v1.json",
        "--state-token-count 8",
        "--history-dropout-curriculum --dual-path-curriculum",
        "--causal-objective-mode specificity_triplet",
        "--causal-repeat-fraction 0.50",
        "augment_recurrent_causal_diagnosis.py",
        "diagnosis_natural_schema",
        "canonical_label_auxiliary_supervision",
        "evaluate_recurrent_diagnosis_c4.py",
        "--expected-adapter-sha256",
        "--expected-recurrent-state-sha256",
        "raw_history_visible_to_decoder",
        "exact_match_is_authoritative",
        "select_latest_valid_distributed_checkpoint.py",
        "stage_distributed_resume_bundle.py",
        "--expected-world-size 4 --max-round 250",
        "--save-steps 25 --recovery-save-steps 25",
        "--probe-expected-cadence",
        "resume_checkpoint_selected=",
        '--resume-from-checkpoint "$resume_checkpoint"',
    ):
        if fragment not in launcher:
            raise ValueError(f"Pilot contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
