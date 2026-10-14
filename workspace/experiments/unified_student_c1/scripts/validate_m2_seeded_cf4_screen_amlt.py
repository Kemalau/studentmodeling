#!/usr/bin/env python3
"""Fail-closed validation for the M2-seeded CF4 screen."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m2_seeded_cf4_screen import (
    BRANCH,
    EVALUATOR_SHA256,
    JOB,
    M2_RUN_DIR,
    RUN_DIR,
    TRAINER_SHA256,
)


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [i for marker in ("description:", "target:") if (i := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}; env = submit.get("env") or {}
    if (
        submit.get("max_attempts") != 2
        or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
        or env.get("BRANCH") != BRANCH
        or str(env.get("CAUSAL_WEIGHT")) != "1.0"
    ):
        raise ValueError("Retry/UAI/branch/weight drift")
    command = command_text(job.get("command")); validate_shell_syntax(command, JOB)
    command = command.replace("$$", "$")
    required = (
        f"export INIT_RUN_DIR={M2_RUN_DIR}", f'export RUN_DIR="{RUN_DIR}"',
        '--init-adapter "$INIT_RUN_DIR/final_adapter"',
        '--init-recurrent-state "$INIT_RUN_DIR/recurrent_state.pt"',
        "initialization=m2_cf1_causal_adapter_and_state",
        "cf4_balanced_paired_state_swap_train_seed20260731_v1.json",
        "--causal-objective-mode donor_directional", "--causal-repeat-fraction 1.0",
        "--diagnostic-max-micro-steps 1000", "--gradient-accumulation-steps 4",
        "--causal-weight \"$CAUSAL_WEIGHT\"", "--behavior-kd-weight 0.5",
        "c4_stage_a_gate_pairs_seed42_n150_v1.json", "--num-pairs 150",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "teacher_cache_train_n12000_top256_fixedu4_v2_20260730",
        TRAINER_SHA256,
        EVALUATOR_SHA256,
    )
    forbidden = (
        "M3_RUN_DIR", "world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730/final_adapter",
        "password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models",
        "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
    )
    if any(value not in command for value in required):
        raise ValueError("M2-seeded screen command drift")
    if any(value.lower() in command.lower() for value in forbidden):
        raise ValueError("Post-M3/secret/local path leaked into M2-seeded screen")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
