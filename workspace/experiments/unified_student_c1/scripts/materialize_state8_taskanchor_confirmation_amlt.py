#!/usr/bin/env python3
"""Materialize full formal and three-seed C4 confirmation for one milestone."""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

import yaml


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
PAIR_HASHES = {
    13: "9856e643848d11a834761e3d28fb078f6eb70fa6c42fc468c3eb1b3a1aafd693",
    42: "edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8",
    97: "15ed710992abd71c37bdfa2246a5a38b5eb422e70ecbfb6da933be013d6d9118",
}


def materialize(template: Path, output: Path, step: int, hashes: dict[str, str]) -> None:
    if step not in {200, 400, 600, 800, 1000}:
        raise ValueError("unsupported milestone")
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("invalid AMLT template")
    common = {
        "_AZUREML_SINGULARITY_JOB_UAI": UAI,
        "VENV_ROOT": "/tmp/studentmodeling-state8-taskanchor-confirm-venv",
        "MILESTONE_STEP": str(step),
        "MILESTONE_ADAPTER_SHA256": hashes["adapter_model"],
        "MILESTONE_STATE_SHA256": hashes["recurrent_state"],
        "MILESTONE_RECEIPT_SHA256": hashes["receipt"],
    }
    setup = [
        "python -m venv --system-site-packages /tmp/studentmodeling-state8-taskanchor-confirm-venv",
        "/tmp/studentmodeling-state8-taskanchor-confirm-venv/bin/python -m pip install --upgrade pip",
        "/tmp/studentmodeling-state8-taskanchor-confirm-venv/bin/python -m pip install -r requirements-feeds.txt",
    ]
    base_job = (config.get("jobs") or [None])[0]
    if not isinstance(base_job, dict):
        raise ValueError("template has no job")
    jobs = []
    for index in range(16):
        job = deepcopy(base_job)
        job.update(
            name=f"state8-taskanchor-confirm-step{step}-formal-shard{index:02d}",
            sku="80G1-A100",
            process_count_per_node=1,
            mpi=False,
            sla_tier="Standard",
            command="set -euo pipefail\nbash scripts/run_state8_taskanchor_confirmation_formal.sh\n",
        )
        job["submit_args"] = {
            "max_attempts": 2,
            "max_run_duration_seconds": 259200,
            "env": {**common, "NUM_SHARDS": "16", "SHARD_INDEX": str(index)},
        }
        jobs.append(job)
    for seed, digest in PAIR_HASHES.items():
        job = deepcopy(base_job)
        job.update(
            name=f"state8-taskanchor-confirm-step{step}-c4-seed{seed}",
            sku="80G1-A100",
            process_count_per_node=1,
            mpi=False,
            sla_tier="Standard",
            command="set -euo pipefail\nbash scripts/run_state8_taskanchor_confirmation_c4.sh\n",
        )
        job["submit_args"] = {
            "max_attempts": 2,
            "max_run_duration_seconds": 259200,
            "env": {**common, "PAIR_SEED": str(seed), "PAIR_MANIFEST_SHA256": digest},
        }
        jobs.append(job)
    config.update(
        description=(
            f"State8 diagnosis-first task-anchor milestone step {step}: full 21676-row "
            "strict-state-only evaluation and 150-pair three-seed answer/diagnosis C4"
        ),
        target={"service": "manifold", "name": "Feeds", "workspace_name": WORKSPACE},
        environment={"image": IMAGE, "setup": setup},
        code={
            "local_dir": "$CONFIG_DIR/..",
            "ignore": [".git/", ".vscode/", "amlt/", "data/", "results/", "__pycache__/", "*.pyc"],
        },
        storage={
            "student_store": {
                "storage_account_name": "csnewsandfeeds4150361735",
                "container_name": "amulet",
                "mount_dir": "/mnt/amulet",
                "is_output": True,
            }
        },
        jobs=jobs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    materialize(
        args.template,
        args.output,
        args.step,
        {
            "adapter_model": args.adapter_sha256,
            "recurrent_state": args.state_sha256,
            "receipt": args.receipt_sha256,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()

