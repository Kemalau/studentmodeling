#!/usr/bin/env python3
"""Fail-closed AMLT validation for State8 semantic causal consolidation."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
COMMON_ENV = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-state8-semantic-causal-venv",
    "REPAIR_ARM": "trajectory01",
    "TRAIN_ENTRYPOINT": "scripts/train_state8_semantic_causal_repair.py",
    "STUDENT_ROOT": "/mnt/amulet/studentmodeling/checkpoints/state8_diagnosis_causal_rescue_seed43_v1_20260806",
    "STUDENT_MODEL_DIR": "/mnt/amulet/studentmodeling/checkpoints/state8_diagnosis_causal_rescue_seed43_v1_20260806/final_adapter",
    "STUDENT_STATE_PATH": "/mnt/amulet/studentmodeling/checkpoints/state8_diagnosis_causal_rescue_seed43_v1_20260806/recurrent_state.pt",
    "STUDENT_RECEIPT": "/mnt/amulet/studentmodeling/checkpoints/state8_diagnosis_causal_rescue_seed43_v1_20260806/completion.json",
    "EXPECTED_STUDENT_ADAPTER_SHA256": "bbd43f677c8fd68b1b5425968747f7ae6e11f55dba6d9a9c0d2332e1609dbb17",
    "EXPECTED_STUDENT_STATE_SHA256": "40a75e9152d054ef3f2d599c72a85cd1e081529ce2be5fa4075abf48de9ec3f4",
    "TRAIN_SEED": "45",
    "STATE8_CAUSAL_LABEL_WEIGHT": "4.0",
    "STATE8_RANDOM_NEUTRAL": "1",
    "STATE8_DIFFERENT_MAX_COSINE": "0.90",
}
JOBS = {
    "state8-semantic-causal-seed45": {
        **COMMON_ENV,
        "RUN_TAG": "semantic_causal_strongbase_v1",
        "STATE8_GEOMETRY_WEIGHT": "0.1",
    },
    "state8-semantic-causal-g005-seed45": {
        **COMMON_ENV,
        "RUN_TAG": "semantic_causal_strongbase_g005_v1",
        "STATE8_GEOMETRY_WEIGHT": "0.05",
    },
    "state8-semantic-causal-g025-seed45": {
        **COMMON_ENV,
        "RUN_TAG": "semantic_causal_strongbase_g025_v1",
        "STATE8_GEOMETRY_WEIGHT": "0.25",
    },
    "state8-eedi-geometry-g005-seed45": {
        **COMMON_ENV,
        "VENV_ROOT": "/tmp/studentmodeling-state8-eedi-semantic-venv",
        "RUN_TAG": "semantic_causal_strongbase_eedi_g005_v1",
        "STATE8_GEOMETRY_WEIGHT": "0.10",
        "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
    },
    "state8-eedi-geometry-g010-seed45": {
        **COMMON_ENV,
        "VENV_ROOT": "/tmp/studentmodeling-state8-eedi-semantic-venv",
        "RUN_TAG": "semantic_causal_strongbase_eedi_g010_v1",
        "STATE8_GEOMETRY_WEIGHT": "0.10",
        "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
    },
    "state8-eedi-geometry-g025-seed45": {
        **COMMON_ENV,
        "VENV_ROOT": "/tmp/studentmodeling-state8-eedi-semantic-venv",
        "RUN_TAG": "semantic_causal_strongbase_eedi_g025_v1",
        "STATE8_GEOMETRY_WEIGHT": "0.10",
        "STATE8_EEDI_GEOMETRY_WEIGHT": "0.25",
    },
    "state8-eedi-geometry-g050-seed45": {
        **COMMON_ENV,
        "VENV_ROOT": "/tmp/studentmodeling-state8-eedi-semantic-venv",
        "RUN_TAG": "semantic_causal_strongbase_eedi_g050_v1",
        "STATE8_GEOMETRY_WEIGHT": "0.10",
        "STATE8_EEDI_GEOMETRY_WEIGHT": "0.50",
    },
}
FILES = {
    "run_state8_stateonly_repair_ab.sh": "2ce887c682ac7d8e9d2ff7c243a177282c985fd935aedbb887a4c308e8e3939a",
    "train_state8_semantic_causal_repair.py": "14db1f4c430c61537a33c0b4617f1734a0bd51501249dfa8f61bc8c0ec01759a",
    "filter_state8_causal_diagnosis.py": "ba04c7d3cd1e74843678d2b3e66a0dea7c0aac647e7b149f18fc1f7b3bebbc58",
    "augment_state8_eedi_semantic_pairs.py": "c9006ac322d075ea1fa35ee9ace3a6561f799bdfea767786a6e459825b0ad46c",
    "train_state8_stateonly_teacher_repair.py": "3f8e05a6b35515c52b53482f894afbaedca6de4ce163a53b0737aa8971637452",
    "train_recurrent_student_m4.py": "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
    "augment_recurrent_causal_same_rule.py": "1ae82541e97353d23b674af80dfb0e2d50cfd92e3a6ee99c2b55f3b0caf1d785",
    "evaluate_recurrent_formal_protocols.py": "262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed",
    "evaluate_recurrent_diagnosis_c4.py": "e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid config")
    return value


def validate(config: dict) -> None:
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
    if code.get("local_dir") is not None and not {
        ".git/", "data/", "results/"
    }.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusion drift")
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    if not names or not names.issubset(JOBS) or len(names) != len(jobs):
        raise ValueError("Job drift")
    commands = []
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku") != "80G4-A100"
            or job.get("process_count_per_node") != -1
            or job.get("mpi") is not True
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}
        if (submit.get("env") or {}) != JOBS[name]:
            raise ValueError("Environment drift")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError("Retry/duration drift")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if "run_state8_stateonly_repair_ab.sh" not in command:
            raise ValueError("Entrypoint drift")
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Matched-arm command drift")

    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
