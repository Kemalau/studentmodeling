#!/usr/bin/env python3
"""Fail-closed validation for the taxonomy-contract high-causal pilot."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "state8-taxonomy-eedi-highcausal-seed51"
ENV = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-state8-taxonomy-eedi-venv",
    "REPAIR_ARM": "trajectory01",
    "TRAIN_ENTRYPOINT": "scripts/train_state8_taxonomy_causal_repair.py",
    "STUDENT_ROOT": "/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1",
    "STUDENT_MODEL_DIR": "/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1/final_adapter",
    "STUDENT_STATE_PATH": "/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1/recurrent_state.pt",
    "STUDENT_RECEIPT": "/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1/completion.json",
    "EXPECTED_STUDENT_ADAPTER_SHA256": "449162955d459e0d60ce3613ca9318ca417eef4986844f37803ac68200a04e69",
    "EXPECTED_STUDENT_STATE_SHA256": "43c16db184b69bf01bd7d63d8b2b9b05ffda215c932376d600a4dc2e35e5e6ad",
    "RUN_TAG": "taxonomy_eedi_highcausal_v1",
    "TRAIN_SEED": "51",
    "STATE8_CAUSAL_LABEL_WEIGHT": "8.0",
    "STATE8_REPAIR_LABEL_WEIGHT": "4.0",
    "STATE8_RANDOM_NEUTRAL": "1",
    "STATE8_GEOMETRY_WEIGHT": "0.5",
    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.25",
    "STATE8_DIFFERENT_MAX_COSINE": "0.75",
    "STATE8_CAUSAL_DIAGNOSIS_ONLY": "1",
    "STATE8_CAUSAL_REPEAT_FRACTION": "1.0",
    "STATE8_CAUSAL_WEIGHT": "4.0",
    "STATE8_CAUSAL_MARGIN": "1.0",
    "STATE8_LAYERWISE_TRANSITION_ONLY": "0",
    "STATE8_UPDATER_ONLY": "0",
}
FILES = {
    "run_state8_stateonly_repair_ab.sh": "2ce887c682ac7d8e9d2ff7c243a177282c985fd935aedbb887a4c308e8e3939a",
    "train_state8_taxonomy_causal_repair.py": "26e62969a03aab89047dd42d5bbe6d144eae9bb77d0fa0b962b0da89b9b75347",
    "state8_taxonomy_contract.py": "1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed",
    "train_state8_semantic_causal_repair.py": "14db1f4c430c61537a33c0b4617f1734a0bd51501249dfa8f61bc8c0ec01759a",
    "evaluate_recurrent_formal_protocols_taxonomy.py": "648baa998672f30652ad3e68f114c2648c5bf839dcddd5b1e0acae47400e5b22",
    "evaluate_recurrent_diagnosis_c4_taxonomy.py": "c3d34960d0f6caff3265b772686bd84222637f628d5fce000215d076c5b2a355",
}


def validate(value: dict) -> None:
    rendered = yaml.safe_dump(value, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(value.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (value.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (value.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = value.get("code") or {}
    if not (
        code.get("local_dir") == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusion drift")
    jobs = value.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G4-A100"
        or job.get("process_count_per_node") != -1
        or job.get("mpi") is not True
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("env") != ENV:
        raise ValueError("Environment drift")
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry/duration drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if "bash scripts/run_state8_stateonly_repair_ab.sh" not in command:
        raise ValueError("Entrypoint drift")
    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    value = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid config")
    validate(value)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
