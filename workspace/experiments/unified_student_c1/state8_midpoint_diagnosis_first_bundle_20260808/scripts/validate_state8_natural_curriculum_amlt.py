#!/usr/bin/env python3
"""Fail-closed preflight for the eight-slot natural curriculum diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "state8-natural-curriculum-seed42"
FILES = {
    "train_recurrent_student_m4.py": "b0a726cdd416af47d18ec15e82a4f4b24bb32707a48fc9ecaf6dd2d05626c9f2",
    "train_recurrent_student_m4_state8_natural_curriculum.py": "503846f21c919143b5b79304eca4c8e5445c506fc76b54035b3a967922141799",
    "state8_natural_curriculum_view.py": "98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682",
    "layerwise_recurrent_student_model.py": "b41fe1f0cfa055b784ccc5f6b2be2c07cfb424d3be6e63e6d1eb2ed061851152",
    "recurrent_student_model.py": "b0a33b50fe5631a9865358664b3c71d45dcbd9dad466e131107dbe3783eedb12",
    "evaluate_recurrent_causal_gate.py": "e0c9fe1361dfda5eddc14abcd275fbe8e2b92f472f9eaf27468b71f800fafccf",
    "evaluate_recurrent_formal_protocols.py": "262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
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
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if not (
        str(code.get("local_dir")) == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code upload exclusions drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job set drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G4-A100"
        or int(job.get("process_count_per_node", 0)) != -1
        or job.get("mpi") is not True
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    required = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "BRANCH": "eedi_dense",
        "LOSS_PROFILE": "state_path",
        "TRAIN_ENTRYPOINT": "scripts/train_recurrent_student_m4_state8_natural_curriculum.py",
        "SEGMENT_ROWS": "1000",
        "LEARNING_RATE": "2e-6",
        "CONSISTENCY_WEIGHT": "0.25",
        "CAUSAL_VARIANT": "same_rule",
        "EXPECTED_CAUSAL_PAIRS": "50",
        "DIAGNOSTIC_STOP_AFTER_ROUND": "3250",
        "MANIFEST": "manifests/layerwise_task_retention_eedi_dense_seed42_v1.json",
        "MANIFEST_SHA256": "cc484d7f7b6db02cfdf1e530866e029a721896d9c57103313e581f504c7bc16e",
    }
    if {key: str(env.get(key)) for key in required} != required:
        raise ValueError("Controlled recipe drift")
    if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("UAI drift")
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry/duration drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    for fragment in (
        "set -euo pipefail",
        "FAILED line=%s rc=%s command=%s",
        "studentmodeling-state8-natural-venv.lock",
        "studentmodeling-state8-natural-venv.complete",
        "pip install -r requirements-feeds.txt",
        "bash scripts/run_state8_natural_curriculum.sh",
    ):
        if fragment not in command:
            raise ValueError(f"Launcher drift: {fragment}")
    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")
    launcher = (root / "run_state8_natural_curriculum.sh").read_text(encoding="utf-8")
    for fragment in (
        "--state-token-count 8 --history-dropout-curriculum --dual-path-curriculum",
        "--online-kd-weight 0.5 --online-kd-temperature 1.0 --online-kd-topk 256",
        "--diagnostic-allow-causal-data-transition",
        "--causal-objective-mode specificity_triplet",
        "--t3-natural-schema",
        "--t3-natural-schema",
    ):
        if fragment not in launcher:
            raise ValueError(f"Launcher invariant drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
