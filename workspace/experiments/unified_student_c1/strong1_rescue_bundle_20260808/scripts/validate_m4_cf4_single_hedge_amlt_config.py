#!/usr/bin/env python3
"""Fail-closed AMLT validation for the single-A100 CF4 queue hedge."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m4_cf4_diagnostic import TRAIN_JSONL_SHA256, TRAIN_MANIFEST, TRAIN_MANIFEST_SHA256
from render_m4_cf4_single_hedge import BRANCH, JOB


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [i for marker in ("description:", "target:") if (i := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts):])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
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
    if len(jobs) != 1 or str(jobs[0].get("name")) != JOB:
        raise ValueError("Job set drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    if (
        submit.get("max_attempts") != 2
        or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
        or env.get("BRANCH") != BRANCH
        or str(env.get("CAUSAL_WEIGHT")) != "1.0"
    ):
        raise ValueError("Retry/UAI/branch drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    required = (
        TRAIN_MANIFEST,
        TRAIN_MANIFEST_SHA256,
        TRAIN_JSONL_SHA256,
        "recurrent_causal_pairs_cf4_balanced_v1",
        "--causal-objective-mode donor_directional",
        "--causal-preference-reduction sum",
        "--causal-repeat-fraction 1.0",
        "--diagnostic-max-micro-steps 1000",
        "--gradient-accumulation-steps 4",
        "optimizer_steps') == 250",
        "causal[:500]",
        "causal[500:]",
        "distributed') is False",
        "c4_stage_a_gate_pairs_seed42_n150_v1.json",
        "--num-pairs 150",
        "world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730",
        "checkpoint-*",
    )
    if any(value not in command for value in required):
        raise ValueError("Command drift")
    forbidden = (
        "--distributed-mpi",
        "--distributed-world-size",
        "--diagnostic-stop-after-round",
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
    )
    if any(value in command.lower() for value in forbidden):
        raise ValueError("Forbidden topology/path/secret")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
