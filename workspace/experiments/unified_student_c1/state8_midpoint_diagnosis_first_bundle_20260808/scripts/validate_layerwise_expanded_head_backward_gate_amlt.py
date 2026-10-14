#!/usr/bin/env python3
"""Fail-closed validator for the three-row expanded-head SDPA backward gate."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "layerwise-expanded-head-backward-549-seed42": (177752, 549, ":233:1"),
    "layerwise-expanded-head-backward-653-seed42": (
        110499,
        653,
        ":0002079:malrule_identity",
    ),
    "layerwise-expanded-head-backward-721-seed42": (
        68865,
        721,
        ":0002130:malrule_identity",
    ),
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
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
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
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
        raise ValueError("Code-upload root drift")
    if not {".git/", "data/", "results/", "amlt/"}.issubset(
        set(code.get("ignore") or [])
    ):
        raise ValueError("Code upload can leak local artifacts")

    jobs = config.get("jobs") or []
    if len(jobs) != 3 or {str(job.get("name")) for job in jobs} != set(JOBS):
        raise ValueError("Job-set drift")
    commands: list[str] = []
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G1-A100"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 1 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry/duration drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        observed = (
            int(env.get("ROW_INDEX", -1)),
            int(env.get("EXPECTED_TOKENS", -1)),
            str(env.get("EXPECTED_SUFFIX")),
        )
        if observed != JOBS[name]:
            raise ValueError(f"Exact-row mapping drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Backward-gate jobs do not share one command")
    if commands[0].strip() != (
        "set -euo pipefail\n"
        "bash scripts/run_layerwise_expanded_head_backward_gate.sh"
    ):
        raise ValueError("Backward-gate launcher drift")

    script = (
        Path(__file__).resolve().parent
        / "run_layerwise_expanded_head_backward_gate.sh"
    ).read_text(encoding="utf-8")
    required = (
        "177752:549:*:233:1",
        "110499:653:*:0002079:malrule_identity",
        "68865:721:*:0002130:malrule_identity",
        "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
        "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
        "b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440",
        "test_verify_sdpa_aligned_mask_equivalence.py",
        "--state-architecture layerwise --layerwise-operation-checkpointing",
        "--layerwise-transition-rank 256",
        "--max-length 1536 --direct-offload-min-tokens 512",
        "--diagnostic-max-micro-steps 1 --diagnostic-real-index \"$ROW_INDEX\"",
        "FINALIZE_EXISTING",
        "gate_scope': 'single_real_row_backward_only'",
        "formal_causal_audit_waived': False",
        "config['planned_causal_exposures'] == 0",
        "config['causal_train_pairs'] == 1051",
        "config['causal_repeat_fraction'] == 0.05",
        "RuntimeError: M4 diagnostic missed causal path:",
        "finalize_gate",
        "history[0]['direct_offload_used'] is True",
        "history[0]['direct_tokens'] == tokens",
        "history[0]['peak_cuda_allocated_gib'] < 79.0",
        "history[0]['peak_cuda_reserved_gib'] < 79.0",
        "expanded_head_backward_receipt.json",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way",
    )
    if any(fragment not in script for fragment in required):
        raise ValueError("Backward-gate script drift")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--distributed-mpi",
        "--resume-from-checkpoint",
    )
    if any(fragment.lower() in script.lower() for fragment in forbidden):
        raise ValueError("Secret/local/distributed path leaked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
