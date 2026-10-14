#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_taskonly_qwen_oneepoch.sh"
IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
SPECS = {
    "qwen3-32b-taskonly-t1-correctness-oneepoch-seed42": ("predict_correctness", "110525", "a98d0f24d47fa0ef1a06fed7946935f6f569ef7eced8a35e2f2289f02ec76046", "t1_correctness"),
    "qwen3-32b-taskonly-t2-response-oneepoch-seed42": ("predict_response", "55569", "e44efac7f183fdb7b5fa4dfe2075e5b10efaa5abf689dc33e0fd718256dfc62e", "t2_response"),
    "qwen3-32b-taskonly-t4-prospective-oneepoch-seed42": ("predict_prospective_error", "78600", "240dc2f1a02bbb446e4d73b0301ee4da7f6716c088746f7ec17bf82934c2aa1e", "t4_prospective"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path) -> None:
    raw=path.read_text(encoding="utf-8")
    if any(value in raw for value in ("<workspace-name>", "<job-name>", "<your-command>", "REQUIRED", "PENDING", "__RUNNER_SHA256__")):
        raise ValueError("Unresolved placeholder")
    cfg=yaml.safe_load(raw)
    if cfg["target"] != {"service":"manifold","name":"Feeds","workspace_name":"CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"}:
        raise ValueError("Target drift")
    if cfg["environment"]["image"] != IMAGE:
        raise ValueError("Image drift")
    if cfg["code"]["local_dir"] != str(ROOT) or "data/" not in cfg["code"]["ignore"] or "results/" not in cfg["code"]["ignore"]:
        raise ValueError("Code upload scope drift")
    jobs=cfg["jobs"]
    if {job["name"] for job in jobs} != set(SPECS):
        raise ValueError("Job matrix drift")
    runner_hash=digest(RUNNER)
    for job in jobs:
        if (job["sku"],job["process_count_per_node"],job["mpi"],job["sla_tier"]) != ("80G1-A100",1,False,"Standard"):
            raise ValueError("Topology drift")
        if runner_hash not in str(job["command"]) or "bash scripts/run_taskonly_qwen_oneepoch.sh" not in str(job["command"]):
            raise ValueError("Runner binding drift")
        task,rows,data_hash,tag=SPECS[job["name"]]
        expected={"_AZUREML_SINGULARITY_JOB_UAI":"rankfun_aml","TASKONLY_TASK":task,"TASKONLY_ROWS":rows,"TASKONLY_DATA_SHA256":data_hash,"TASKONLY_RUN_TAG":tag}
        env={str(k):str(v) for k,v in job["submit_args"]["env"].items()}
        if env != expected or job["submit_args"]["max_attempts"] != 2 or job["submit_args"]["max_run_duration_seconds"] != 172800:
            raise ValueError("Environment or runtime policy drift")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args(); validate(args.config); print("valid=true")


if __name__ == "__main__": main()
