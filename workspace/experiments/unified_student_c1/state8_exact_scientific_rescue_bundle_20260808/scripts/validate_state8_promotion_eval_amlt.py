#!/usr/bin/env python3
"""Fail-closed AMLT preflight for state8 full-T3 and multiseed-C4 gates."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
T3_JOBS = {f"state8-promotion-t3-shard{index}": index for index in range(4)}
T3_JOBS.update({f"state8-balanced-promotion-t3-shard{index}": index for index in range(4)})
T3_JOBS.update({f"state8-highkd-promotion-t3-shard{index}": index for index in range(4)})
C4_JOBS = {
    "state8-promotion-c4-seed13": (13, "9856e643848d11a834761e3d28fb078f6eb70fa6c42fc468c3eb1b3a1aafd693"),
    "state8-promotion-c4-seed42": (42, "edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8"),
    "state8-promotion-c4-seed97": (97, "15ed710992abd71c37bdfa2246a5a38b5eb422e70ecbfb6da933be013d6d9118"),
}
C4_JOBS.update({
    name.replace("state8-promotion", "state8-balanced-promotion"): value
    for name, value in tuple(C4_JOBS.items())
})
C4_JOBS.update({
    name.replace("state8-promotion", "state8-highkd-promotion"): value
    for name, value in tuple(C4_JOBS.items())
    if name.startswith("state8-promotion-")
})
FILES = {
    "evaluate_recurrent_formal_protocols.py": "262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed",
    "evaluate_state8_t3_resumable.py": "50e33ccff0caa0aee62b2a201d35095dc8bc365614200018904d7aa698c737ed",
    "t3_resumable_progress.py": "892b984179b49523141e8950af603b88f363349a297ee396be01acdad7f6ba4c",
    "evaluate_recurrent_causal_gate.py": "e0c9fe1361dfda5eddc14abcd275fbe8e2b92f472f9eaf27468b71f800fafccf",
    "merge_state8_promotion_eval.py": "3ba79da2575e0d57b9709489ec6d4d35459b41195282c9c8a47db13e2377b9b7",
    "state8_natural_curriculum_view.py": "98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
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
        raise ValueError("Code exclusions drift")
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    full_names = set(T3_JOBS) | set(C4_JOBS)
    if names != (expected_jobs or full_names):
        raise ValueError(f"Job set drift: {sorted(names)}")
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku") != "80G1-A100"
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("mpi") is not False
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        expected_variant = (
            "balanced" if name.startswith("state8-balanced-")
            else "highkd" if name.startswith("state8-highkd-")
            else "t3_heavy"
        )
        if env.get("PROMOTION_VARIANT") != expected_variant:
            raise ValueError(f"Promotion variant drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 43200:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command"))
        normalized_command = command.replace("$$", "$")
        validate_shell_syntax(command, name)
        for fragment in (
            "set -euo pipefail",
            "FAILED line=%s rc=%s command=%s",
            "state8_qualified_teacher_curriculum_v1_immutable",
            "state8_qualified_teacher_balanced_curriculum_v1_immutable",
            "state8_qualified_teacher_curriculum_highkd_v1_immutable",
            "80G1-A100" if False else "Qwen3-32B",
            "metrics/completion.json",
            "immutable_receipt.json",
            "receipt['micro_steps']==5000",
            "state8_qualified_teacher_curriculum_immutable_v1",
            "state8_qualified_teacher_balanced_curriculum_immutable_v1",
            "state8_qualified_teacher_curriculum_highkd_immutable_v1",
            "teacher_adapter_sha256",
            "qualified_teacher_receipt.json",
            "receipt['sha256']",
        ):
            if fragment not in command:
                raise ValueError(f"Launcher drift {name}: {fragment}")
        if name in T3_JOBS:
            if str(env.get("SHARD_INDEX")) != str(T3_JOBS[name]):
                raise ValueError(f"T3 shard drift: {name}")
            for fragment in (
                "--tasks diagnose_misconception --sources eedi_mining malrule",
                "--shard-index",
                "--num-shards 4",
                "--generation-progress-dir",
                "--generation-progress-key",
                "evaluate_state8_t3_resumable.py",
                "state8_qualified_teacher_${PROMOTION_VARIANT}_v1/t3_shard${SHARD_INDEX}_v1",
                "--save-unredacted-predictions --t3-natural-schema",
                "unsharded_filtered_rows']==1494",
            ):
                    if fragment not in normalized_command:
                        raise ValueError(f"T3 protocol drift {name}: {fragment}")
        else:
            seed, manifest_sha = C4_JOBS[name]
            if str(env.get("PAIR_SEED")) != str(seed):
                raise ValueError(f"C4 seed drift: {name}")
            if env.get("PAIR_MANIFEST_SHA256") != manifest_sha:
                raise ValueError(f"C4 manifest drift: {name}")
            for fragment in (
                "--num-pairs 150",
                "--seed",
                "--progress-dir",
                "--progress-key",
            ):
                if fragment not in command:
                    raise ValueError(f"C4 protocol drift {name}: {fragment}")
    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
