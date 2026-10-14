#!/usr/bin/env python3
"""Fail-closed validation for matched State8 high-causal Feeds arms."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
COMMON = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-state8-high-causal-venv",
    "REPAIR_ARM": "trajectory01",
    "TRAIN_ENTRYPOINT": "scripts/train_state8_semantic_causal_repair.py",
    "STUDENT_ROOT": "/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1",
    "STUDENT_MODEL_DIR": "/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1/final_adapter",
    "STUDENT_STATE_PATH": "/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1/recurrent_state.pt",
    "STUDENT_RECEIPT": "/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed45_trajectory01_semantic_causal_strongbase_v1/completion.json",
    "EXPECTED_STUDENT_ADAPTER_SHA256": "449162955d459e0d60ce3613ca9318ca417eef4986844f37803ac68200a04e69",
    "EXPECTED_STUDENT_STATE_SHA256": "43c16db184b69bf01bd7d63d8b2b9b05ffda215c932376d600a4dc2e35e5e6ad",
    "STATE8_CAUSAL_LABEL_WEIGHT": "8.0",
    "STATE8_REPAIR_LABEL_WEIGHT": "4.0",
    "STATE8_RANDOM_NEUTRAL": "1",
    "STATE8_GEOMETRY_WEIGHT": "1.0",
    "STATE8_DIFFERENT_MAX_COSINE": "0.75",
    "STATE8_CAUSAL_DIAGNOSIS_ONLY": "1",
    "STATE8_CAUSAL_REPEAT_FRACTION": "1.0",
    "STATE8_CAUSAL_WEIGHT": "8.0",
    "STATE8_CAUSAL_MARGIN": "1.0",
}
EXPECTED = {
    "state8-highcausal-joint-seed47": {
        **COMMON,
        "RUN_TAG": "highcausal_joint_v1",
        "TRAIN_SEED": "47",
        "STATE8_LAYERWISE_TRANSITION_ONLY": "0",
        "STATE8_UPDATER_ONLY": "0",
    },
    "state8-highcausal-updater-seed48": {
        **COMMON,
        "RUN_TAG": "highcausal_updater_only_v1",
        "TRAIN_SEED": "48",
        "STATE8_LAYERWISE_TRANSITION_ONLY": "1",
        "STATE8_UPDATER_ONLY": "1",
    },
}
FILES = {
    "run_state8_stateonly_repair_ab.sh": "2ce887c682ac7d8e9d2ff7c243a177282c985fd935aedbb887a4c308e8e3939a",
    "train_state8_semantic_causal_repair.py": "14db1f4c430c61537a33c0b4617f1734a0bd51501249dfa8f61bc8c0ec01759a",
    "filter_state8_causal_diagnosis.py": "ba04c7d3cd1e74843678d2b3e66a0dea7c0aac647e7b149f18fc1f7b3bebbc58",
    "train_state8_stateonly_teacher_repair.py": "3f8e05a6b35515c52b53482f894afbaedca6de4ce163a53b0737aa8971637452",
    "train_recurrent_student_m4.py": "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
    "augment_recurrent_causal_same_rule.py": "1ae82541e97353d23b674af80dfb0e2d50cfd92e3a6ee99c2b55f3b0caf1d785",
    "augment_recurrent_causal_diagnosis.py": "b74684001e54bf0c9b023540c76a269708f1e8fa529cfec703a4439288590926",
    "subset_state8_causal_twins.py": "fcca21bd73ad8ddd83d1cc9e308183b6b6c751b97537ed3d3d35838567ef87cb",
    "evaluate_recurrent_formal_protocols.py": "262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed",
    "evaluate_recurrent_diagnosis_c4.py": "e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664",
}


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
        code.get("local_dir") == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusion drift")
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != set(EXPECTED):
        raise ValueError("Job set drift")
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
        if submit.get("env") != EXPECTED[name]:
            raise ValueError(f"Environment drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if "run_state8_stateonly_repair_ab.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
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
