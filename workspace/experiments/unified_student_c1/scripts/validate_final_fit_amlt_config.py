#!/usr/bin/env python3
"""Fail-closed validator for post-selection full-corpus recurrent final fitting."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
ARMS = ("generation_only", "generation_kd")
PROJECTIONS = ("identity", "trainable")
CACHE_JOBS = {f"finalfit-cache-rank{rank}-seed42": rank for rank in range(4)}
TRAIN_JOBS = {
    f"finalfit-{arm.replace('_', '-')}-{projection}-seed42": (arm, projection)
    for arm in ARMS
    for projection in PROJECTIONS
}
TRAIN_JOBS["finalfit-generation-kd-trainable-seed42-16rank"] = (
    "generation_kd",
    "trainable",
)
EXPECTED_JOBS = set(CACHE_JOBS) | set(TRAIN_JOBS)
HASHES = {
        "train": "865c12219deb23cf9db145c797244969721fce80323050dcd987b33d96cbfb1b",
    "distributed": "4b825fa7e7f03de5ceb60e3a812eafab79ba47b07bc9ca8c11274c45f24cd178",
    "recurrent": "9e0431dccb524f67b401f7ea911e11c96fc3061432ffdc73e0cac2bbc2acc5a0",
    "cache_builder": "d52b5f17ffbbd033ce2ba27877dbedf9abdaffa4aaaffe8194cbe7bd5a5b7f08",
    "cache_merge": "8bd00421608fe2c31df67ce624d2452183461877f9aa07a1e85d8552bd72c468",
    "cache_loader": "04f057df2ccdc174e3094bab1c79d9bdbee4c9c67657615f1e49c3e68903cc4b",
}


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [
        index
        for marker in ("description:", "target:")
        if (index := text.find(marker)) >= 0
    ]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def selected_train_job(arm: str, projection: str) -> str:
    if arm not in ARMS or projection not in PROJECTIONS:
        raise ValueError(f"Unknown final-fit selection: {arm}/{projection}")
    return f"finalfit-{arm.replace('_', '-')}-{projection}-seed42"


def validate(config: dict, selected_names: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b|PLACEHOLDER", rendered):
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
    jobs = config.get("jobs") or []
    expected = EXPECTED_JOBS if selected_names is None else selected_names
    names = {str(job.get("name")) for job in jobs}
    if names != expected or len(jobs) != len(expected):
        raise ValueError(f"Job set drift: expected={sorted(expected)} got={sorted(names)}")
    for job in jobs:
        name = str(job["name"])
        if name not in EXPECTED_JOBS:
            raise ValueError(f"Unknown final-fit job: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift in {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != UAI:
            raise ValueError(f"UAI drift in {name}")
        if job.get("sla_tier") != "Standard":
            raise ValueError(f"SLA drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        common = (
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl",
            "aeabd3b4dfef3c4d92ac68583879c91448ef90f58304cedf32e470b586ca4542",
            "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd",
        )
        if name in CACHE_JOBS:
            if job.get("sku") != "80G1-A100" or job.get("mpi") is not False:
                raise ValueError(f"Cache GPU/MPI drift in {name}")
            if int(job.get("process_count_per_node", 0)) != 1:
                raise ValueError(f"Cache process-count drift in {name}")
            rank = CACHE_JOBS[name]
            if str(env.get("PARTITION_INDEX")) != str(rank):
                raise ValueError(f"Cache partition drift in {name}")
            required = (
                *common,
                HASHES["cache_builder"],
                "--partition-count 4",
                '--order-seed 42',
                "--top-k 256",
                "--shard-size 512",
            )
        else:
            is_16rank = name.endswith("-16rank")
            expected_sku = "2x80G8-A100-IB-NvLink" if is_16rank else "80G8-A100-IB-NvLink"
            if job.get("sku") != expected_sku or job.get("mpi") is not True:
                raise ValueError(f"Train GPU/MPI drift in {name}")
            if int(job.get("process_count_per_node", 0)) != -1:
                raise ValueError(f"Train process-count drift in {name}")
            arm, projection = TRAIN_JOBS[name]
            if env.get("ARM") != arm or env.get("PROJECTION_MODE") != projection:
                raise ValueError(f"Train selection-label drift in {name}")
            expected_world = "16" if is_16rank else "8"
            expected_accumulation = "1" if is_16rank else "2"
            if str(env.get("DISTRIBUTED_WORLD_SIZE")) != expected_world or str(
                env.get("GRADIENT_ACCUMULATION_STEPS")
            ) != expected_accumulation:
                raise ValueError(f"Train topology-label drift in {name}")
            required = (
                *common,
                HASHES["train"],
                HASHES["distributed"],
                HASHES["recurrent"],
                HASHES["cache_loader"],
                HASHES["cache_merge"],
                "PILOT_CAUSAL_REPEAT",
                "PILOT_CAUSAL_WEIGHT",
                "PILOT_NORMALIZE_CAUSAL",
                "PILOT_CAUSAL_PREFERENCE_REDUCTION",
                "PILOT_CAUSAL_DATA_MODE",
                "'causal_data_mode': os.environ['PILOT_CAUSAL_DATA_MODE']",
                "'causal_train_sha256': config['causal_train_sha256']",
                "'causal_valid_sha256': config['causal_valid_sha256']",
                "causal_args=(--causal-weight",
                "--normalize-causal-exposure",
                "'raw4': (4.0, False, 0.0)",
                "'mean_normalized': (1.0, True, 0.0)",
                "'raw8': (8.0, False, 0.0)",
                "'mean2_normalized': (2.0, True, 0.0)",
                "'repeat100_w1': (1.0, False, 1.0)",
                "'repeat100_w4': (4.0, False, 1.0)",
                "'repeat100_sum_w1': (1.0, False, 1.0)",
                "'repeat100_sum_w4': (4.0, False, 1.0)",
                "'bootstrap_sum_w1': (1.0, False, 1.0)",
                "'bootstrap_sum_w4': (4.0, False, 1.0)",
                "'bootstrap_native25_sum_w1': (1.0, False, 1.0)",
                "'bootstrap_native25_sum_w4': (4.0, False, 1.0)",
                "'bootstrap_balanced_sum_w1': (1.0, False, 1.0)",
                "'bootstrap_balanced_sum_w4': (4.0, False, 1.0)",
                "'bootstrap_native25_balanced_sum_w1': (1.0, False, 1.0)",
                "'bootstrap_native25_balanced_sum_w4': (4.0, False, 1.0)",
                "'donor_directional_sum_w1': (1.0, False, 1.0)",
                "'donor_directional_sum_w4': (4.0, False, 1.0)",
                "'cf4_bootstrap_symmetric_w1': (1.0, False, 1.0)",
                "'cf4_donor_directional_w1': (1.0, False, 1.0)",
                "cf4_balanced_paired_state_swap_train_seed20260731_v1.json",
                "30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766",
                "final_fit_causal_repeat_fraction",
                "final_fit_causal_preference_reduction",
                "final_fit_causal_objective_mode",
                "PILOT_CAUSAL_OBJECTIVE_MODE",
                "C4_D_specific",
                "GRADIENT_ACCUMULATION_STEPS",
                "--gradient-accumulation-steps",
                "--causal-preference-reduction",
                "--causal-objective-mode",
                "--distributed-mpi",
                "DISTRIBUTED_WORLD_SIZE",
                "--distributed-world-size",
                "--lr-scheduler cosine",
                "--warmup-ratio 0.03",
                "--learning-rate 5e-6",
                "--save-steps 2000",
                "causal_pairs_seen') == config.get('planned_causal_exposures')",
                "final_fit_complete",
            )
            pilot_receipt_copies = (
                'cp "$$PILOT_RECEIPT" "$$AMLT_OUTPUT_DIR/metrics/pilot_receipt.json"',
                'cp "$PILOT_RECEIPT" "$AMLT_OUTPUT_DIR/metrics/pilot_receipt.json"',
            )
            if not any(value in command for value in pilot_receipt_copies):
                raise ValueError(f"Pilot-receipt copy drift in {name}")
            repeat_forms = (
                '--causal-repeat-fraction "$$PILOT_CAUSAL_REPEAT"',
                '--causal-repeat-fraction "$PILOT_CAUSAL_REPEAT"',
            )
            if not any(value in command for value in repeat_forms):
                raise ValueError(f"Causal-repeat command drift in {name}")
            if arm == "generation_kd" and "--behavior-kd-weight 0.5" not in command:
                raise ValueError(f"KD path missing in {name}")
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        forbidden = (
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Secret/local/forbidden path drift in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load_config(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
