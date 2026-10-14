#!/usr/bin/env python3
"""Fail-closed preflight for resilient step-2000 answer/diagnosis C4 jobs."""

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
ROOT = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_"
    "20260805_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4/"
    "checkpoint-2000"
)
ADAPTER = "a80766e8dae4881fc8b4fb8dcebd14b4e95d059bf3246b120fd050c856c4d587"
STATE = "9088dbc32437b1feb5b27bf1a4dd6502bff99b15d608c10977feb6f5cef10426"
C4 = "7eef90e079d698c94f1f8655a1df3fe6bd93cedbb175ba4e7082a49bb8069c8b"
ANSWER_C4 = "69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52"
ANSWER = "state8-exact-midpoint-step2000-answer-c4-resumable"
SHARDS = {
    f"state8-exact-midpoint-step2000-diagnosis-c4-shard{start:02d}": start
    for start in (0, 10, 20, 30, 40)
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
    expected = {ANSWER, *SHARDS}
    observed = [str(job.get("name")) for job in jobs]
    if not observed or len(observed) != len(set(observed)) or not set(observed) <= expected:
        raise ValueError("Resilient C4 job set drift")
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
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry/duration drift: {name}")
        env = submit.get("env") or {}
        required = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "STATE8_EXACT_CHECKPOINT_ROOT": ROOT,
            "STATE8_EXACT_ADAPTER_SHA256": ADAPTER,
            "STATE8_EXACT_STATE_SHA256": STATE,
            "EXACT_DIAGNOSIS_C4_SHA256": C4,
            "STATE8_EXACT_ANSWER_C4_SHA256": ANSWER_C4,
        }
        for key, value in required.items():
            if env.get(key) != value:
                raise ValueError(f"Environment drift {name}: {key}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if name == ANSWER:
            if "run_state8_exact_midpoint_answer_c4_resumable.sh" not in command:
                raise ValueError("Answer-C4 entrypoint drift")
            progress = str(env.get("STATE8_EXACT_ANSWER_PROGRESS_DIR") or "")
            key = str(env.get("STATE8_EXACT_ANSWER_PROGRESS_KEY") or "")
            if not progress.startswith("/mnt/amulet/studentmodeling/eval_progress/"):
                raise ValueError("Answer-C4 progress is not remote")
            if ADAPTER[:8] not in key or STATE[:8] not in key or "pairs50-seed42" not in key:
                raise ValueError("Answer-C4 progress key is not checkpoint-bound")
        else:
            if "run_state8_exact_midpoint_diagnosis_c4_shard.sh" not in command:
                raise ValueError(f"Diagnosis-C4 entrypoint drift: {name}")
            if int(env.get("PAIR_START", -1)) != SHARDS[name]:
                raise ValueError(f"Diagnosis-C4 pair start drift: {name}")
            if int(env.get("PAIR_COUNT", -1)) != 10:
                raise ValueError(f"Diagnosis-C4 pair count drift: {name}")

    root = Path(__file__).resolve().parent
    answer_source = (root / "run_state8_exact_midpoint_answer_c4_resumable.sh").read_text()
    diagnosis_source = (root / "run_state8_exact_midpoint_diagnosis_c4_shard.sh").read_text()
    for fragment in (
        '--progress-dir "$STATE8_EXACT_ANSWER_PROGRESS_DIR"',
        '--progress-key "$STATE8_EXACT_ANSWER_PROGRESS_KEY"',
        "--num-pairs 50 --allow-manifest-prefix --seed 42",
        ADAPTER,
        STATE,
        'STATE8_EXACT_ANSWER_C4_SHA256',
    ):
        if fragment not in answer_source:
            raise ValueError(f"Answer-C4 launcher drift: {fragment}")
    for fragment in (
        '--pair-start "$PAIR_START"',
        '--pair-count "$PAIR_COUNT"',
        "--max-new-tokens 160",
        "--num-pairs 50",
        ADAPTER,
        STATE,
    ):
        if fragment not in diagnosis_source:
            raise ValueError(f"Diagnosis-C4 launcher drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
