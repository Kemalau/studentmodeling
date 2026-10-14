#!/usr/bin/env python3
"""Fail-closed validation for the State8 interleaved retention Feeds job."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_state8_interleaved_retention_branch.sh"
MANIFEST = ROOT / "manifests/state8_interleaved_retention_seed8401_v1.json"
IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
EXPECTED_ENV = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-state8-interleaved-retention-venv",
    "REPAIR_BRANCH": "early_label_high",
    "REPAIR_SEED": "8401",
    "SOURCE_CHECKPOINT_ARM": "curriculum_current_query",
    "SOURCE_CHECKPOINT_STEP": "3000",
    "SOURCE_CHECKPOINT_ROOT": "/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_current_query_curriculum_20260812/checkpoint-3000",
    "SOURCE_ANCHOR_MODE": "full_mse1",
    "EXPECTED_STUDENT_ADAPTER_SHA256": "566bd6a488523bd59610767e019c53c5f1609b760a1ab1d1fcf6cbaabc95cabd",
    "EXPECTED_STUDENT_STATE_SHA256": "5e0b0d40ee5dee2429f641270cdbb83e66b96172cd53e3336e84ac9b6564ac73",
    "REPAIR_REMOTE_ROOT": "/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401",
    "STATE8_REPAIR_MANIFEST": "manifests/state8_interleaved_retention_seed8401_v1.json",
    "STATE8_REPAIR_MANIFEST_SHA256": "bb502f9bbbcdd8beb92d474da16ed8f803eb388840107ba1df1841fd4fcff85c",
    "STATE8_T6_ANCHOR_SCHEDULE": "1.0,1.0,1.0",
    "STATE8_EEDI_CAUSAL_TWINS": "200",
    "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
    "STATE8_CAUSAL_WEIGHT": "0.5",
    "STATE8_CAUSAL_MARGIN": "0.5",
    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
    "STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT": "0.15",
    "STATE8_CAUSAL_TEACHER_COSINE_WEIGHT": "1.0",
    "STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT": "1.0",
    "STATE8_EARLY_LABEL_BINDING_WEIGHT": "0.5",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    if any(token in raw for token in ("<workspace-name>", "<job-name>", "<your-command>", "REQUIRED", "PENDING")):
        raise ValueError("Unresolved placeholder")
    config = yaml.safe_load(raw)
    target = config.get("target") or {}
    template_target = {
        "service": "manifold",
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity",
    }
    dumped_target = {
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
    }
    if target not in (template_target, dumped_target, {"service": "sing", **dumped_target}):
        raise ValueError("Feeds target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    code = config.get("code") or {}
    if "local_dir" in code and code.get("local_dir") != str(ROOT):
        raise ValueError("Code root drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Remote storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Expected one independent job")
    job = jobs[0]
    if job.get("name") != "state8-interleaved-retention-early-label-high-seed8401":
        raise ValueError("Job name drift")
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
        "80G1-A100", 1, False, "Standard"
    ):
        raise ValueError("GPU topology or SLA drift")
    if digest(RUNNER) not in str(job.get("command") or ""):
        raise ValueError("Runner hash binding drift")
    env = {str(key): str(value) for key, value in ((job.get("submit_args") or {}).get("env") or {}).items()}
    if env != EXPECTED_ENV:
        raise ValueError("Environment contract drift")
    if digest(MANIFEST) != EXPECTED_ENV["STATE8_REPAIR_MANIFEST_SHA256"]:
        raise ValueError("Interleaved manifest hash drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print("valid=true")


if __name__ == "__main__":
    main()
