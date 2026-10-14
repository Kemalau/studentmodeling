#!/usr/bin/env python3
"""Fail-closed validation of the four-way Eedi specialist recovery config."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path, selected: set[str] | None = None) -> None:
    raw = path.read_text(encoding="utf-8")
    if any(token in raw for token in ("<workspace-name>", "<job-name>", "<your-command>", "REQUIRED", "PENDING")):
        raise ValueError("Unresolved placeholder")
    config = yaml.safe_load(raw)
    target = config["target"]
    if target.get("name") != "Feeds" or target.get("service") not in {None, "manifold", "sing"}:
        raise ValueError("Feeds target drift")
    workspace = target.get("workspace_name")
    if workspace not in {
        "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity",
        "CS-NewsAndFeeds-Singularity",
    }:
        raise ValueError("Workspace drift")
    rendered_resource_group = target.get("workspace_resource_group")
    if rendered_resource_group not in {None, "rg-cs-newsandfeeds-singularity"}:
        raise ValueError("Workspace resource-group drift")
    if config["environment"]["image"] != IMAGE:
        raise ValueError("Image drift")
    code = config["code"]
    if "local_dir" in code and code["local_dir"] != str(ROOT):
        raise ValueError("Code scope drift")
    if code.get("ignore") != [".git/", ".vscode/", "amlt/", "data/", "results/", "__pycache__/", "*.pyc"]:
        raise ValueError("Code ignore drift")
    jobs = config["jobs"]
    expected = {f"qwen3-32b-t3-eedi-taskonly-formal-shard{index:02d}" for index in range(4)}
    names = {job["name"] for job in jobs}
    if names != (selected or expected):
        raise ValueError(f"Job matrix drift: {names}")
    runner_hash = digest(ROOT / "scripts/run_t3_eedi_taskonly_qwen_formal_shard.sh")
    for job in jobs:
        if (job["sku"], job["process_count_per_node"], job["mpi"], job["sla_tier"]) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError("Job topology drift")
        index = int(job["name"][-2:])
        command = str(job["command"])
        if runner_hash not in command or "bash scripts/run_t3_eedi_taskonly_qwen_formal_shard.sh" not in command:
            raise ValueError("Runner hash binding drift")
        submit = job["submit_args"]
        if submit["max_attempts"] != 2 or submit["max_run_duration_seconds"] != 86400:
            raise ValueError("Runtime policy drift")
        env = submit["env"]
        if env.get("_AZUREML_SINGULARITY_JOB_UAI", "").split("/")[-1] != "rankfun_aml":
            raise ValueError("UAI drift")
        if env.get("SHARD_INDEX") != str(index):
            raise ValueError("Shard coordinate drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(args.config, set(args.jobs) if args.jobs else None)
    print("valid=true")


if __name__ == "__main__":
    main()
