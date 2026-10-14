#!/usr/bin/env python3
"""Fail-closed validation for the T3-Eedi task-only Qwen formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
ADAPTER_SHA256 = "329e95ad02a10cd9541f81ce3f220aaa521369f5761659a146fc2c54ad859bb2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    if any(value in raw for value in ("<workspace-name>", "<job-name>", "<your-command>", "REQUIRED", "PENDING")):
        raise ValueError("Unresolved placeholder")
    config = yaml.safe_load(raw)
    if config["target"] != {
        "service": "manifold",
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity",
    }:
        raise ValueError("Feeds target drift")
    if config["environment"]["image"] != IMAGE:
        raise ValueError("Image drift")
    if config["code"] != {
        "local_dir": str(ROOT),
        "ignore": [".git/", ".vscode/", "amlt/", "data/", "results/", "__pycache__/", "*.pyc"],
    }:
        raise ValueError("Code upload scope drift")
    if config["storage"]["student_store"] != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Remote storage drift")
    jobs = config["jobs"]
    if len(jobs) != 1:
        raise ValueError("Expected one independent evaluation job")
    job = jobs[0]
    if (job["name"], job["sku"], job["process_count_per_node"], job["mpi"], job["sla_tier"]) != (
        "qwen3-32b-t3-eedi-taskonly-formal-open-seed42", "80G1-A100", 1, False, "Standard"
    ):
        raise ValueError("Job topology drift")
    command = str(job["command"])
    required = (
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl",
        "/mnt/amulet/studentmodeling/adapters/qwen3_32b_eedi_t3_qlora_1epoch_len1536/final_adapter",
        "--tasks diagnose_misconception --sources eedi_mining",
        "--save-unredacted-predictions --t3-llm-judge",
        "--judge-model gpt-5.5",
        "group['n']==894",
        ADAPTER_SHA256,
        digest(ROOT / "scripts/evaluate_formal_protocols.py"),
        digest(ROOT / "scripts/copy_model_parallel.py"),
        digest(ROOT / "requirements-feeds.txt"),
    )
    if any(value not in command for value in required):
        raise ValueError("Evaluation protocol or hash binding drift")
    submit = job["submit_args"]
    if submit["max_attempts"] != 2 or submit["max_run_duration_seconds"] != 86400:
        raise ValueError("Runtime policy drift")
    if submit["env"] != {"_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml"}:
        raise ValueError("UAI drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print("valid=true")


if __name__ == "__main__":
    main()
