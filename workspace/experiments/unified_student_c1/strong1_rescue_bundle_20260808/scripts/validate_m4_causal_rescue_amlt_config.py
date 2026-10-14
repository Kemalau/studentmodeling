#!/usr/bin/env python3
"""Fail-closed validation for rendered causal-rescue M4 training configs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m4_causal_rescue import BRANCHES, TRAIN_SHA256


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [
        index
        for marker in ("code:", "description:", "target:")
        if (index := text.find(marker)) >= 0
    ]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
    return value


def validate(config: dict, selected: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    allowed = {spec["job"]: (branch, spec) for branch, spec in BRANCHES.items()}
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    expected = names if selected is None else selected
    if not expected or not expected.issubset(allowed) or names != expected:
        raise ValueError(f"Job set drift: {sorted(names)}")
    for job in jobs:
        name = str(job["name"])
        branch, spec = allowed[name]
        if job.get("sku") != "80G4-A100" or job.get("mpi") is not True:
            raise ValueError(f"GPU/MPI drift in {name}")
        if int(job.get("process_count_per_node", 0)) != -1:
            raise ValueError(f"Process count drift in {name}")
        if job.get("sla_tier") != "Standard":
            raise ValueError(f"SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift in {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift in {name}")
        if (
            env.get("BRANCH") != branch
            or str(env.get("CAUSAL_WEIGHT")) != spec["weight"]
            or str(env.get("NORMALIZE_CAUSAL")) != spec["normalize"]
        ):
            raise ValueError(f"Branch environment drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        dense = spec["repeat"] == "1.0"
        cf4 = spec.get("data_mode") == "cf4_balanced"
        checkpoint_round = 250 if cf4 else 500
        unique_pairs = 1051 if cf4 else 1726
        required = (
            TRAIN_SHA256,
            f"checkpoint-round-{checkpoint_round}",
            f"receipt.get('round') == {checkpoint_round}",
            "checkpoint-round-*",
            "--distributed-world-size 4",
            "completion.get('micro_steps') == 12000",
            (
                "completion.get('causal_pairs_seen') == 12000"
                if dense
                else "completion.get('causal_pairs_seen') == 1726"
            ),
            "causal_rescue/selected_seed42.json",
            "evaluate_recurrent_causal_gate.py",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "--num-pairs 150",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            f"--causal-preference-reduction {spec['preference_reduction']}",
            f"--causal-objective-mode {spec['objective_mode']}",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        if "--diagnostic-stop-after-round" in command:
            raise ValueError(f"Rescue job is still diagnostic in {name}")
        resume_forms = (
            '--resume-from-checkpoint "$$resume_checkpoint"',
            '--resume-from-checkpoint "$resume_checkpoint"',
        )
        if not any(value in command for value in resume_forms):
            raise ValueError(f"Resume command drift in {name}")
        if dense:
            checkpoint_source_mode = (
                "donor_directional"
                if spec["objective_mode"] == "donor_directional"
                else (
                    "bootstrap_then_directional"
                    if spec["objective_mode"] == "directional"
                    else "bootstrap_then_symmetric"
                )
            )
            dense_required = (
                "--causal-repeat-fraction 1.0",
                f"completion.get('unique_causal_pairs_seen') == {unique_pairs}",
                "final_fit_causal_repeat_fraction",
                "final_fit_causal_preference_reduction",
                f"config.get('causal_objective_mode') == '{spec.get('checkpoint_objective_mode', checkpoint_source_mode)}'",
            )
            if any(value not in command for value in dense_required):
                raise ValueError(f"Dense rehearsal drift in {name}")
        if cf4:
            cf4_required = (
                "recurrent_causal_pairs_cf4_balanced_v1",
                "cf4_balanced_paired_state_swap_train_seed20260731_v1.json",
                "30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766",
                "'causal_data_mode': 'cf4_balanced'",
            )
            if any(value not in command for value in cf4_required):
                raise ValueError(f"CF4 data drift in {name}")
        forbidden = (
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Forbidden local/secret path in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
