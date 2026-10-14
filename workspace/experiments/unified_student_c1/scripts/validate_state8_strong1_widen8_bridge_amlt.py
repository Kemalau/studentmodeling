#!/usr/bin/env python3
"""Fail-closed preflight for the strong-one-slot to State8 bridge."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:  # package-style unit tests
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
PREFIX = (
    "/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/"
    "eedi_full_open_t3/state_path/lr5e-7_start17000_stop4500_"
    "08945e930bc8b9a55e57a41d4656612e3dfce99b8544c88068110a74a1a22da9_"
    "b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_"
    "b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a/"
    "checkpoint-round-4500"
)
EXPECTED = {
    "state8-strong1-widen8-bridge-lr5e7-seed58": ("5e-7", "58", "strong1_widen8_lr5e7_seed58"),
    "state8-strong1-widen8-bridge-lr1e6-seed59": ("1e-6", "59", "strong1_widen8_lr1e6_seed59"),
}
SOURCE_WORLD4_SHA256 = "b545496b52c8aff82196c8ac917178b110e304b1ae4880ed24ca039cff626638"


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
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
    jobs = config.get("jobs") or []
    job_names = {str(job.get("name")) for job in jobs}
    if not job_names or not job_names.issubset(EXPECTED) or len(job_names) != len(jobs):
        raise ValueError("Job drift")
    for job in jobs:
        name = str(job["name"])
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G2-A100", -1, True, "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 43200:
            raise ValueError("Retry/duration drift")
        env = submit.get("env") or {}
        lr, seed, suffix = EXPECTED[name]
        expected = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "STATE8_REPLAY_VARIANT": "t3_heavy",
            "STATE8_KD_VARIANT": "highkd",
            "STATE8_CURRICULUM_NAMESPACE": "taxonomy_semantic_v1",
            "TRAIN_ENTRYPOINT": "scripts/train_state8_taxonomy_teacher_curriculum.py",
            "FORMAL_EVALUATOR": "scripts/evaluate_recurrent_formal_protocols_taxonomy.py",
            "STATE8_CURRICULUM_LABEL_WEIGHT": "4.0",
            "STATE8_SAVE_STEPS": "10",
            "STATE8_DISTRIBUTED_WORLD_SIZE": "2",
            "STATE8_STUDENT_ROOT": PREFIX,
            "STATE8_STUDENT_MODEL_DIR": PREFIX + "/adapter",
            "STATE8_STUDENT_STATE_PATH": PREFIX + "/recurrent_state.pt",
            "STATE8_STUDENT_RECEIPT": PREFIX + "/checkpoint_completion.json",
            "STATE8_STUDENT_INIT_CAPACITY": "1",
            "STATE8_STUDENT_INIT_TAG": "strong1_round4500",
            "STATE8_STUDENT_INIT_ADAPTER_SHA256": "827532591bbcc5587bc78a0108463dac43ceab8c702ee6f60e120c68a2afe482",
            "STATE8_STUDENT_INIT_STATE_SHA256": "701ed585383d919bdfe9956ddbbfe870b2dcf323a3ae4adde257cf2365e419da",
            "STATE8_LEARNING_RATE": lr,
            "STATE8_RUN_SUFFIX": suffix,
            "TRAIN_SEED": seed,
        }
        for key, value in expected.items():
            if str(env.get(key)) != value:
                raise ValueError(f"Environment drift: {name}/{key}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if "run_state8_qualified_teacher_curriculum_world2.sh" not in command:
            raise ValueError("Entrypoint drift")
        root = Path(__file__).resolve().parent
        wrapper = (root / "run_state8_qualified_teacher_curriculum_world2.sh").read_text()
        for fragment in (
            SOURCE_WORLD4_SHA256,
            "c8149a6555bbbc56eaa93b3d3324146e30abd97d8586cc199a85d81d07f722c1",
            "render_state8_world2_launcher.py",
            'bash -n "$rendered_launcher"',
        ):
            if fragment not in wrapper:
                raise ValueError(f"World2 wrapper drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
