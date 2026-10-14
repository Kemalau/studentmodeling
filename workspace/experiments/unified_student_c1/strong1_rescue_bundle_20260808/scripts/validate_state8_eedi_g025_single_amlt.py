#!/usr/bin/env python3
"""Fail-closed validation for the single-A100 EEDI semantic screen."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
FILES = {
    "run_state8_stateonly_repair_ab.sh": "2ce887c682ac7d8e9d2ff7c243a177282c985fd935aedbb887a4c308e8e3939a",
    "train_state8_semantic_causal_repair.py": "5eba0a8e0ed37d1525a31c5c0e72ab547d18ecb78be462dc3ffa0230bb30f8a2",
    "augment_state8_eedi_semantic_pairs.py": "c9006ac322d075ea1fa35ee9ace3a6561f799bdfea767786a6e459825b0ad46c",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Invalid config")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    code = config.get("code") or {}
    if not (
        code.get("local_dir") == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != "state8-eedi-g025-single-seed46":
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or job.get("process_count_per_node") != 1
        or job.get("mpi") is not False
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("Topology drift")
    env = (job.get("submit_args") or {}).get("env") or {}
    expected = {
            "REQUESTED_WORLD_SIZE": "1",
            "GRADIENT_ACCUMULATION_STEPS": "4",
        "TRAIN_SEED": "46",
        "STATE8_EEDI_GEOMETRY_WEIGHT": "0.25",
        "STATE8_GEOMETRY_WEIGHT": "0.10",
        "STATE8_RANDOM_NEUTRAL": "1",
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    }
    for key, value in expected.items():
        if env.get(key) != value:
            raise ValueError(f"Environment drift: {key}")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, str(job["name"]))
    if "run_state8_stateonly_repair_ab.sh" not in command:
        raise ValueError("Entrypoint drift")
    root = Path(__file__).resolve().parent
    for name, expected_hash in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected_hash:
            raise ValueError(f"Source hash drift for {name}: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
