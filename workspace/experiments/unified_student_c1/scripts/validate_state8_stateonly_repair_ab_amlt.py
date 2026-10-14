#!/usr/bin/env python3
"""Fail-closed AMLT validation for the State8 state-only repair A/B."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "state8-repair-logits-seed44": "logits",
    "state8-repair-trajectory01-seed44": "trajectory01",
}
FILES = {
    "run_state8_stateonly_repair_ab.sh": "2ce887c682ac7d8e9d2ff7c243a177282c985fd935aedbb887a4c308e8e3939a",
    "train_state8_stateonly_teacher_repair.py": "3f8e05a6b35515c52b53482f894afbaedca6de4ce163a53b0737aa8971637452",
    "train_recurrent_student_m4.py": "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
    "state8_natural_curriculum_view.py": "98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682",
    "evaluate_recurrent_formal_protocols.py": "262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed",
    "evaluate_recurrent_diagnosis_c4.py": "e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664",
    "select_latest_valid_distributed_checkpoint.py": "033de45a0d16eb5d36fe9f8552bdebad7e7c8db2cc5d4cdd2a8e1084773aaeb2",
    "stage_distributed_resume_bundle.py": "c89b26b4e5b69d60774e2fceed57e7c7a635d4b5d403e7e0f1e952127ada6708",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid config")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if not (
        code.get("local_dir") == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusion drift")
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    if names != (expected_jobs or set(JOBS)):
        raise ValueError(f"Job drift: {sorted(names)}")
    commands = []
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku") != "80G4-A100"
            or job.get("process_count_per_node") != -1
            or job.get("mpi") is not True
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if env.get("REPAIR_ARM") != JOBS[name]:
            raise ValueError(f"Repair arm drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if "run_state8_stateonly_repair_ab.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("A/B arms differ beyond the arm environment")

    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")
    launcher = (root / "run_state8_stateonly_repair_ab.sh").read_text(encoding="utf-8")
    for fragment in (
        "STATE8_REPAIR_TRAJECTORY_WEIGHT=0.0",
        "STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1",
        "STATE8_REPAIR_LABEL_WEIGHT",
        "state8_stateonly_repair_seed${TRAIN_SEED}_${REPAIR_ARM}_${RUN_TAG}",
        "state8_stateonly_repair_seed44_v1.json",
        "--history-dropout-curriculum --dual-path-curriculum",
        "--online-kd-weight 1.0",
        "--causal-objective-mode specificity_triplet",
        "--state-token-count 8",
        "--tasks diagnose_misconception",
        "--num-pairs 50",
        "raw_history_visible_to_decoder':False",
        "all(bool(x['history_dropped']) for x in h)",
    ):
        if fragment not in launcher:
            raise ValueError(f"Launcher drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
