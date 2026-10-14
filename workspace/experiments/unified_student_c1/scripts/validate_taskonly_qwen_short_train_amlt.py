#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
JOBS = {
    "qwen3-32b-taskonly-t3-full-oneepoch-seed42": ("diagnose_misconception", 6776, "7971db483e791236b179ac21b2f646383ced8427e87fb0fbf6f98473c37504ab"),
    "qwen3-32b-taskonly-t5-persistence-oneepoch-seed42": ("predict_error_persistence", 8375, "0db13b88287d5afc05a53b41f99d83f9e2a812904c4170dd4c168f90c512aa4d"),
}


def validate(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    if any(value in raw for value in ("<workspace-name>", "<job-name>", "<your-command>", "REQUIRED", "PENDING")):
        raise ValueError("Unresolved placeholder")
    cfg = yaml.safe_load(raw)
    if cfg["target"] != {"service": "manifold", "name": "Feeds", "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"}:
        raise ValueError("Target drift")
    if cfg["environment"]["image"] != IMAGE:
        raise ValueError("Image drift")
    if cfg["code"]["local_dir"] != str(ROOT) or "results/" not in cfg["code"]["ignore"] or "data/" not in cfg["code"]["ignore"]:
        raise ValueError("Code upload scope drift")
    jobs = cfg["jobs"]
    if {job["name"] for job in jobs} != set(JOBS):
        raise ValueError("Job matrix drift")
    for job in jobs:
        task, rows, digest = JOBS[job["name"]]
        if (job["sku"], job["process_count_per_node"], job["mpi"], job["sla_tier"]) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError("Topology drift")
        command = str(job["command"])
        required = (
            f"--task-filter {task}", f"--max-train-samples {rows}",
            "--max-steps -1 --num-train-epochs 1",
            "--per-device-train-batch-size 1 --gradient-accumulation-steps 4",
            "--learning-rate 1e-4 --lora-rank 8 --lora-alpha 16 --load-4bit",
            "--state-token-alias '<|vision_pad|>'", "--save-final-only --seed 42",
            digest, "test ! -e \"$$RUN_DIR/completion.json\"",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Protocol drift: {job['name']}")
        if job["submit_args"] != {"max_attempts": 2, "max_run_duration_seconds": 86400, "env": {"_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml"}}:
            raise ValueError("Submission policy drift")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(); validate(args.config); print("valid=true")


if __name__ == "__main__": main()
