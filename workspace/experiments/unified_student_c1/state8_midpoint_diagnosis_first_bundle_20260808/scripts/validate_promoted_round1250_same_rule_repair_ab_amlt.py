#!/usr/bin/env python3
"""Fail-closed validation for the round-1250 same-rule repair A/B."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
NAMES = {
    "promoted-round1250-repair-base-seed42": "base",
    "promoted-round1250-repair-same-rule-seed42": "same_rule",
}
COMMON = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-round1250-samerule-ab-venv",
    "BRANCH": "eedi_full_open_t3",
    "LOSS_PROFILE": "state_path",
    "TRAIN_ENTRYPOINT": "scripts/train_recurrent_student_m4_open_t3.py",
    "SAME_RULE_AUGMENTER_SHA256": "d0f5eb225e771a745874ab9a6c829b8cd4ac2bd71754ae96f8c8007498bc1ff2",
    "SEGMENT_ROWS": "1000",
    "LEARNING_RATE": "5e-7",
    "EXPECTED_CAUSAL_PAIRS": "50",
    "DIAGNOSTIC_START_GLOBAL_POSITION": "20000",
    "DIAGNOSTIC_STOP_AFTER_ROUND": "5250",
    "MANIFEST": "manifests/layerwise_promoted_round1250_same_rule_repair_balanced_seed42_v1.json",
    "MANIFEST_SHA256": "f612eec632b185679a8c746034927047e5c9ac4b808238ee2e1d753fda89cd36",
    "WARM_START_CHECKPOINT_DIR": "/mnt/amulet/studentmodeling/eval_artifacts/promoted_round1250_frozen_v1/checkpoint-round-1250",
    "WARM_START_EXPECTED_ROUND": "1250",
    "WARM_START_EXPECTED_WORLD_SIZE": "16",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
    return value


def validate(config: dict, *, selected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|<describe your experiment>", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    expected_names = set(NAMES) if selected_jobs is None else set(selected_jobs)
    if not expected_names or not expected_names <= set(NAMES):
        raise ValueError("Unknown selected job")
    jobs = config.get("jobs") or []
    if len(jobs) != len(expected_names) or {job.get("name") for job in jobs} != expected_names:
        raise ValueError("Job-set drift")
    commands = []
    observed_envs = {}
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G4-A100"
            or job.get("mpi") is not True
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        env = {str(k): str(v) for k, v in (submit.get("env") or {}).items()}
        expected = {**COMMON, "CAUSAL_VARIANT": NAMES[name]}
        if env != expected:
            raise ValueError(f"Controlled arm drift: {name}")
        observed_envs[name] = env
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("A/B jobs do not share one command")
    if set(observed_envs) == set(NAMES):
        left = copy.deepcopy(observed_envs["promoted-round1250-repair-base-seed42"])
        right = copy.deepcopy(observed_envs["promoted-round1250-repair-same-rule-seed42"])
        left.pop("CAUSAL_VARIANT"); right.pop("CAUSAL_VARIANT")
        if left != right:
            raise ValueError("A/B controls differ beyond causal variant")
    command = commands[0].replace("$$", "$")
    required = (
        "FAILED line=%s rc=%s command=%s",
        "run_layerwise_task_retention_rehearsal.sh",
        "f46c3906365085114cd8649832b448fcfd5551139878fc8f042eafad0592de48",
        "d0f5eb225e771a745874ab9a6c829b8cd4ac2bd71754ae96f8c8007498bc1ff2",
        "f612eec632b185679a8c746034927047e5c9ac4b808238ee2e1d753fda89cd36",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("Runner/provenance hash drift")
    forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models")
    if any(fragment.lower() in command.lower() for fragment in forbidden):
        raise ValueError("Secret/local model path leaked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", choices=tuple(NAMES))
    args = parser.parse_args()
    validate(load(args.config), selected_jobs=set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
