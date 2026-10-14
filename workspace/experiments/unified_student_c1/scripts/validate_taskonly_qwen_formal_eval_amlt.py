#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
SPECS = {
    "qwen3-32b-taskonly-t1-formal-full-seed42": ("T1", "predict_correctness", "taskonly_qwen_t1_correctness_oneepoch_seed42_v1", "10913", "110525"),
    "qwen3-32b-taskonly-t2-formal-full-seed42": ("T2", "predict_response", "taskonly_qwen_t2_response_oneepoch_seed42_v1", "3693", "55569"),
    "qwen3-32b-taskonly-t3-formal-full-seed42": ("T3", "diagnose_misconception", "taskonly_qwen_t3_full_oneepoch_seed42_v1", "1494", "6776"),
    "qwen3-32b-taskonly-t4-formal-full-seed42": ("T4", "predict_prospective_error", "taskonly_qwen_t4_prospective_oneepoch_seed42_v1", "4600", "78600"),
    "qwen3-32b-taskonly-t5-formal-full-seed42": ("T5", "predict_error_persistence", "taskonly_qwen_t5_persistence_oneepoch_seed42_v1", "976", "8375"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path, selected: set[str] | None = None) -> None:
    raw = path.read_text(encoding="utf-8")
    if any(value in raw for value in ("<workspace-name>", "<job-name>", "<your-command>", "REQUIRED", "PENDING")):
        raise ValueError("Unresolved placeholder")
    cfg = yaml.safe_load(raw)
    source_target = {"service": "manifold", "name": "Feeds", "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"}
    dump_target = {"name": "Feeds", "workspace_name": "CS-NewsAndFeeds-Singularity", "workspace_resource_group": "rg-cs-newsandfeeds-singularity"}
    if cfg["target"] not in (source_target, dump_target):
        raise ValueError("Target drift")
    if cfg["environment"]["image"] != IMAGE:
        raise ValueError("Image drift")
    expected_ignore = [".git/", ".vscode/", "amlt/", "data/", "results/", "__pycache__/", "*.pyc"]
    code = cfg["code"]
    if code.get("ignore") != expected_ignore:
        raise ValueError("Code upload scope drift")
    if "local_dir" in code and code["local_dir"] != str(ROOT):
        raise ValueError("Source code path drift")
    if "storage_id" in code and code["storage_id"] != "default":
        raise ValueError("Dump code storage drift")
    storage = cfg["storage"]["student_store"]
    if storage != {"storage_account_name": "csnewsandfeeds4150361735", "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True}:
        raise ValueError("Remote storage drift")
    jobs = cfg["jobs"]
    expected_names = selected or set(SPECS)
    if {job["name"] for job in jobs} != expected_names:
        raise ValueError("Job matrix drift")
    runner_hash = digest(ROOT / "scripts/run_taskonly_qwen_formal_eval.sh")
    for job in jobs:
        if (job["sku"], job["process_count_per_node"], job["mpi"], job["sla_tier"]) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError("Topology drift")
        command = str(job["command"])
        if runner_hash not in command or "bash scripts/run_taskonly_qwen_formal_eval.sh" not in command:
            raise ValueError("Runner hash binding drift")
        public, task, adapter, rows, train_rows = SPECS[job["name"]]
        expected_env = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "TASKONLY_PUBLIC_TASK": public,
            "TASKONLY_TASK": task,
            "TASKONLY_ADAPTER_TAG": adapter,
            "TASKONLY_EXPECTED_ROWS": rows,
            "TASKONLY_TRAIN_ROWS": train_rows,
        }
        env = {str(key): str(value) for key, value in job["submit_args"]["env"].items()}
        if env != expected_env:
            raise ValueError("Task/adapter/denominator binding drift")
        if job["submit_args"]["max_attempts"] != 2 or job["submit_args"]["max_run_duration_seconds"] != 172800:
            raise ValueError("Runtime policy drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    selected = set(args.jobs) if args.jobs else None
    if selected is not None and not selected.issubset(SPECS):
        raise ValueError("Unknown selected job")
    validate(args.config, selected)
    print("valid=true")


if __name__ == "__main__":
    main()
