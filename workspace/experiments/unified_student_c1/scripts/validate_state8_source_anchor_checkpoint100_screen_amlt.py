#!/usr/bin/env python3
"""Fail-closed validation for the immutable checkpoint-100 screen."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
RUNNER = "run_state8_source_anchor_checkpoint100_screen.sh"
EXPECTED = {
    "state8-source-anchor-cp100-full-mse1": {
        "SOURCE_ANCHOR_ARM": "full_mse1",
        "SOURCE_ANCHOR_ADAPTER_SHA256": "e4f86a02f699aa1025bba58e21dd1f447a6f79cc60ff27f63238f5f51c50a89f",
        "SOURCE_ANCHOR_STATE_SHA256": "e7820634e691020ca1251cf387f3af23f8b26156171bcf9262a06eaa0bfe23e4",
        "SOURCE_ANCHOR_RECEIPT_SHA256": "1177eb0c627c85889989e846c828cf457879731af725c3025fd647086d93ff17",
    },
    "state8-source-anchor-cp100-updater-mse1": {
        "SOURCE_ANCHOR_ARM": "updater_mse1",
        "SOURCE_ANCHOR_ADAPTER_SHA256": "73b6633759febc6ccddac8e81d8715b512a86c3410cf6d2f160b48610d59c558",
        "SOURCE_ANCHOR_STATE_SHA256": "6234a567af99f079d6fd9c223fabc1e1c0c8a4b8c220c7ff512ff1e8d2e19e72",
        "SOURCE_ANCHOR_RECEIPT_SHA256": "e79470cd6309841f5a240ffa1e416daeccaf47f8a9955de46f142e232ddf6bfd",
    },
    "state8-source-anchor-cp100-full-mse2": {
        "SOURCE_ANCHOR_ARM": "full_mse2",
        "SOURCE_ANCHOR_ADAPTER_SHA256": "97a096c90cc5724c875d903f8c88a4f1d1b510ff01e9385ef9bddcd69d83f973",
        "SOURCE_ANCHOR_STATE_SHA256": "b5e8f2fd8bb5534b941bdc97c956de6860ce3def6f77fd0ced93c464d39ae339",
        "SOURCE_ANCHOR_RECEIPT_SHA256": "eac1af6e3f5a0ae6832ba7d29d6e3689b6bcfddb211abfe6bf0884153d70ea4c",
    },
}


def validate(path: Path, selected_jobs: set[str] | None = None) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
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
    jobs = config.get("jobs") or []
    names = set(EXPECTED) if selected_jobs is None else set(selected_jobs)
    if not names or not names <= set(EXPECTED):
        raise ValueError("Invalid selected jobs")
    if {str(job.get("name")) for job in jobs} != names:
        raise ValueError("Job matrix drift")
    runner_hash = hashlib.sha256((Path(__file__).parent / RUNNER).read_bytes()).hexdigest()
    for job in jobs:
        name = str(job["name"])
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry/duration drift: {name}")
        env = submit.get("env") or {}
        expected_env = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-source-anchor-cp100-screen-venv",
            **EXPECTED[name],
        }
        if env != expected_env:
            raise ValueError(f"Environment/artifact drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if runner_hash not in command or f"bash scripts/{RUNNER}" not in command:
            raise ValueError(f"Runner binding drift: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", action="append")
    args = parser.parse_args()
    validate(args.config, set(args.job) if args.job else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
