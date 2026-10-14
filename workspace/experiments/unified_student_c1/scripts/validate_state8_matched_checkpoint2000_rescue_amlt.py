#!/usr/bin/env python3
"""Fail-closed validation for checkpoint-2000 causal rescue jobs."""

from __future__ import annotations

import argparse
import hashlib
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
ROOT = "/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/checkpoint-2000"
HASHES = {
    "adapter": "95a36705a12a166be19db4134000546beb7e6dda9caf22670cc1a9f7d9844c73",
    "state": "48af05a9e162c8beabc838548887f2c8bb6136297d4ad5d64899492f1ab2414b",
}
JOBS = {
    "state8-matched-cp2000-rescue-semantic-preserve": "semantic_preserve",
    "state8-matched-cp2000-rescue-causal-taskanchor": "causal_taskanchor",
}
FILES = {
    "run_state8_matched_checkpoint2000_rescue.sh": "c04d4059c8f94a4cb3b1146f2ef470c8943acf5b44e8e665c422265e45bd29ab",
    "run_state8_stateonly_repair_ab.sh": "2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7",
    "select_latest_valid_single_checkpoint.py": "224736e342bb55bd54ad284eb0907a880dccf9bb906e4d22d71b91edb618d243",
    "stage_distributed_resume_bundle.py": "c89b26b4e5b69d60774e2fceed57e7c7a635d4b5d403e7e0f1e952127ada6708",
    "train_recurrent_student_m4.py": "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
    "train_state8_diagnosis_first_causal_repair.py": "7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216",
    "train_state8_diagnosis_first_task_anchored_causal_repair.py": "eb9837127af9fe3761f6fa57b78a3e9ba3373a223706f0a4921749eab61e35a3",
    "evaluate_recurrent_causal_gate.py": "69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52",
}


def validate(path: Path, expected_jobs: set[str] | None = None) -> None:
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
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if not (code.get("local_dir") == "$CONFIG_DIR/.." or code.get("storage_id") == "default"):
        raise ValueError("Code root drift")
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    if names != (expected_jobs or set(JOBS)):
        raise ValueError(f"Job drift: {names}")
    for job in jobs:
        name = str(job["name"]); env = (job.get("submit_args") or {}).get("env") or {}
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError(f"Topology drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if env.get("MATCHED_CHECKPOINT_ROOT") != ROOT or env.get("STATE8_RESCUE_MODE") != JOBS[name]:
            raise ValueError(f"Recipe drift: {name}")
        if env.get("EXPECTED_STUDENT_ADAPTER_SHA256") != HASHES["adapter"] or env.get("EXPECTED_STUDENT_STATE_SHA256") != HASHES["state"]:
            raise ValueError(f"Artifact binding drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        if "run_state8_matched_checkpoint2000_rescue.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift: {name}={observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", action="append")
    args = parser.parse_args()
    validate(args.config, set(args.job) if args.job else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
