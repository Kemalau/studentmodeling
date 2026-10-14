#!/usr/bin/env python3
"""Fail-closed validation for immutable one-GPU State8 diagnosis-C4 shards."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
EXPECTED = {f"state8-c4-shard{start:02d}": start for start in (0, 10, 20, 30, 40)}


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
        raise ValueError("Shard job drift")
    starts = []
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku") != "80G1-A100"
            or job.get("process_count_per_node") != 1
            or job.get("mpi") is not False
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        start = int(env.get("PAIR_START", -1))
        if start != EXPECTED[name] or int(env.get("PAIR_COUNT", -1)) != 10:
            raise ValueError(f"Pair range drift: {name}")
        starts.append(start)
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if "run_state8_diagnosis_c4_shard.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    if sorted(starts) != [0, 10, 20, 30, 40]:
        raise ValueError("Non-disjoint pair coverage")
    launcher = (Path(__file__).parent / "run_state8_diagnosis_c4_shard.sh").read_text()
    for fragment in (
        "--num-pairs 50",
        '--pair-start "$PAIR_START"',
        '--pair-count "$PAIR_COUNT"',
        "449162955d459e0d60ce3613ca9318ca417eef4986844f37803ac68200a04e69",
        "43c16db184b69bf01bd7d63d8b2b9b05ffda215c932376d600a4dc2e35e5e6ad",
        "--max-new-tokens 160",
    ):
        if fragment not in launcher:
            raise ValueError(f"Launcher drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
