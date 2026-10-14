#!/usr/bin/env python3
"""Fail-closed AMLT preflight for the matched checkpoint-1200 screen."""

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
    "state8-matched-checkpoint1200-t3": "scripts/run_state8_matched_checkpoint1200_t3.sh",
    "state8-matched-checkpoint1200-c4": "scripts/run_state8_matched_checkpoint1200_c4.sh",
}
HASHES = {
    "scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py": "a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9",
    "scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py": "660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca",
    "scripts/evaluate_recurrent_causal_gate.py": "69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52",
    "scripts/build_recurrent_evaluation_manifest.py": "d335b0b97471683eb1f317f21af6f0dd36ccca0fe6a8e495838112785ea85a8e",
    "scripts/copy_model_parallel.py": "62b37ab70bc2ffc81c8314ca69215da7facdf18dba3f023d83096d1c3526ea4c",
    "manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json": "edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path, project: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    code = config.get("code") or {}
    if code.get("local_dir") is not None:
        if code.get("local_dir") != "$CONFIG_DIR/..":
            raise ValueError("Code root drift")
    elif code.get("storage_id") != "default":
        raise ValueError("Rendered code storage drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    names = [str(job.get("name")) for job in jobs]
    if (
        not names
        or len(names) != len(set(names))
        or not set(names).issubset(JOBS)
    ):
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError("Retry/duration drift")
        if submit.get("env") != {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-matched-checkpoint1200-screen-venv",
        }:
            raise ValueError("UAI/environment drift")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        if f"bash {JOBS[name]}" not in command:
            raise ValueError("Entrypoint drift")
    for relative, expected in HASHES.items():
        actual = digest(project / relative)
        if actual != expected:
            raise ValueError(f"Hash drift {relative}: {actual}")
    for launcher in JOBS.values():
        text = (project / launcher).read_text(encoding="utf-8")
        for fragment in (
            "checkpoint-1200", "strict_state_only", "raw_history_visible_to_decoder",
            "27fcdc000fcf0d410004aad84f78b314fc9bc7d227385aa62c67f9eabeedc356",
            "7a99e2f15379549d3caef6f0d764b97716e143c4881e5dec4e2a54caf184d12e",
        ):
            if fragment not in text:
                raise ValueError(f"Launcher contract drift: {launcher}: {fragment}")
        if any(x in text for x in ("optimizer.step()", "--resume-from-checkpoint", "--init-adapter")):
            raise ValueError(f"Read-only screen mutates weights: {launcher}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    validate(args.config, args.project)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
