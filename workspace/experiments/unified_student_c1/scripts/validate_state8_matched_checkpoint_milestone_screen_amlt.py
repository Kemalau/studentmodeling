#!/usr/bin/env python3
"""Fail-closed validation for a hash-bound two-job matched milestone screen."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
PREFIX = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_"
    "taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/checkpoint-"
)
PLACEHOLDERS = {"HASH_ADAPTER_SHA256", "HASH_STATE_SHA256", "HASH_RECEIPT_SHA256"}


def expected(step: int) -> dict[str, str]:
    return {
        f"state8-matched-checkpoint{step}-t3": "scripts/run_state8_matched_checkpoint_t3.sh",
        f"state8-matched-checkpoint{step}-c4": "scripts/run_state8_matched_checkpoint_milestone_c4.sh",
    }


def validate(path: Path, *, step: int, allow_template: bool = False) -> None:
    if step not in {3000, 4000}:
        raise ValueError("Unsupported milestone step")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if any(value in rendered for value in PLACEHOLDERS) and not allow_template:
        raise ValueError("Unresolved hash placeholder")
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if code.get("local_dir") is not None:
        if code.get("local_dir") != "$CONFIG_DIR/.." or path.parent.name != "amlt":
            raise ValueError("Code root drift")
    elif code.get("storage_id") != "default":
        raise ValueError("Rendered code storage drift")
    jobs = config.get("jobs") or []
    launchers = expected(step)
    names = {str(job.get("name")) for job in jobs}
    if names != set(launchers) or len(names) != len(jobs):
        raise ValueError("Job set drift")
    root = f"{PREFIX}{step}"
    project = Path(__file__).resolve().parents[1]
    for job in jobs:
        name = str(job["name"]); submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError(f"Topology drift {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry/duration drift {name}")
        required = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-matched-milestone-screen-venv",
            "STATE8_MATCHED_CHECKPOINT_ROOT": root,
            "STATE8_MATCHED_CHECKPOINT_STEP": str(step),
        }
        if any(env.get(key) != value for key, value in required.items()):
            raise ValueError(f"Environment drift {name}")
        for key in ("STATE8_MATCHED_ADAPTER_SHA256", "STATE8_MATCHED_STATE_SHA256", "MATCHED_CHECKPOINT_RECEIPT_SHA256"):
            value = str(env.get(key, ""))
            if value in PLACEHOLDERS and allow_template:
                continue
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"Digest drift {name}/{key}")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        launcher = project / launchers[name]
        if f"bash {launchers[name]}" not in command or hashlib.sha256(launcher.read_bytes()).hexdigest() not in command:
            raise ValueError(f"Entrypoint/hash drift {name}")
    if any(value in rendered for value in ("optimizer.step()", "--resume-from-checkpoint", "--init-adapter")):
        raise ValueError("Read-only screen mutates weights")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--allow-template", action="store_true")
    args = parser.parse_args()
    validate(args.config, step=args.step, allow_template=args.allow_template)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
