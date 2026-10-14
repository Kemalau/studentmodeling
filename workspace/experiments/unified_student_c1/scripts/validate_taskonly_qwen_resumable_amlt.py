#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
SPECS = {
    "qwen3-32b-taskonly-t1-resumable-seed42": ("predict_correctness", "110525", "a98d0f24d47fa0ef1a06fed7946935f6f569ef7eced8a35e2f2289f02ec76046", "t1_correctness"),
    "qwen3-32b-taskonly-t2-resumable-seed42": ("predict_response", "55569", "e44efac7f183fdb7b5fa4dfe2075e5b10efaa5abf689dc33e0fd718256dfc62e", "t2_response"),
    "qwen3-32b-taskonly-t4-resumable-seed42": ("predict_prospective_error", "78600", "240dc2f1a02bbb446e4d73b0301ee4da7f6716c088746f7ec17bf82934c2aa1e", "t4_prospective"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [
        index
        for marker in ("code:", "description:", "target:")
        if (index := text.find(marker)) >= 0
    ]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(path: Path, selected_jobs: set[str] | None = None) -> None:
    raw = path.read_text(encoding="utf-8")
    if re.search(
        r"<workspace-name>|<job-name>|<your-command>|\bREQUIRED\b|\bPENDING\b|"
        r"msrcommonacr\.azurecr\.io/ml-common-py3\.12:latest",
        raw,
    ):
        raise ValueError("Unresolved/forbidden value")
    cfg = load(path)
    if not target_matches(cfg.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if cfg["environment"]["image"] != IMAGE:
        raise ValueError("Image drift")
    code = cfg.get("code") or {}
    source_code = code.get("local_dir") == str(ROOT)
    rendered_code = code.get("local_dir") is None and code.get("storage_id") == "default"
    if not (source_code or rendered_code):
        raise ValueError("Upload scope drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Upload exclusions drift")
    storage = (cfg.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    expected_jobs = set(SPECS) if selected_jobs is None else set(selected_jobs)
    if not expected_jobs or not expected_jobs <= set(SPECS):
        raise ValueError("Unknown selected job")
    if set(job["name"] for job in cfg["jobs"]) != expected_jobs:
        raise ValueError("Job matrix drift")
    hashes = {
        "run_taskonly_qwen_resumable.sh": digest(ROOT / "scripts/run_taskonly_qwen_resumable.sh"),
        "train_c1_lora.py": digest(ROOT / "scripts/train_c1_lora.py"),
        "validate_taskonly_qwen_checkpoint.py": digest(ROOT / "scripts/validate_taskonly_qwen_checkpoint.py"),
    }
    for job in cfg["jobs"]:
        if (job["sku"], job["process_count_per_node"], job["mpi"], job["sla_tier"]) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError("Topology drift")
        command = command_text(job["command"])
        validate_shell_syntax(command, str(job["name"]))
        if any(value not in command for value in hashes.values()) or "bash scripts/run_taskonly_qwen_resumable.sh" not in command:
            raise ValueError("Hash binding drift")
        task, rows, data_hash, tag = SPECS[job["name"]]
        expected = {"_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml", "TASKONLY_TASK": task, "TASKONLY_ROWS": rows, "TASKONLY_DATA_SHA256": data_hash, "TASKONLY_RUN_TAG": tag}
        env = {str(key): str(value) for key, value in job["submit_args"]["env"].items()}
        if env != expected or job["submit_args"]["max_attempts"] != 2 or job["submit_args"]["max_run_duration_seconds"] != 172800:
            raise ValueError("Environment/runtime drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", choices=tuple(SPECS))
    args = parser.parse_args()
    validate(args.config, set(args.jobs) if args.jobs else None)
    print("valid=true")


if __name__ == "__main__":
    main()
