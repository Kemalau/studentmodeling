#!/usr/bin/env python3
"""Bind one read-only dual diagnosis-C4 job to each causal-teacher cp20 snapshot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECS = ((8201, "causal_teacher_low"), (8202, "causal_teacher_high"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(template: Path, receipt_dir: Path, output: Path) -> None:
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Template job matrix drift")
    base = jobs[0]
    runner = Path(__file__).with_name("run_state8_repair_dual_c4_quickscreen.sh")
    runner_sha = digest(runner)
    rendered_jobs = []
    for seed, branch in SPECS:
        receipt = receipt_dir / f"seed{seed}.{branch}.checkpoint-20.completion.json"
        value = json.loads(receipt.read_text(encoding="utf-8"))
        if value.get("status") != "complete" or int(value.get("step", -1)) != 20:
            raise ValueError(f"Incomplete checkpoint receipt: {receipt}")
        hashes = value.get("sha256") or {}
        if set(hashes) != {"adapter_model", "recurrent_state"}:
            raise ValueError("Checkpoint artifact hash schema drift")
        if any(len(str(item)) != 64 for item in (*hashes.values(), digest(receipt))):
            raise ValueError("Invalid checkpoint SHA256")
        job = copy.deepcopy(base)
        job["name"] = f"state8-ct-{branch.replace('_', '-')}-seed{seed}-cp20-dual-c4"
        job["command"] = (
            "set -euo pipefail\n"
            f"test \"$$(sha256sum scripts/{runner.name} | cut -d' ' -f1)\" = {runner_sha}\n"
            f"bash scripts/{runner.name}\n"
        )
        submit = job.setdefault("submit_args", {})
        submit["max_attempts"] = 2
        submit["max_run_duration_seconds"] = 21600
        submit["env"] = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-repair-dual-c4-venv",
            "REPAIR_SEED": str(seed),
            "REPAIR_BRANCH": branch,
            "CHECKPOINT_STEP": "20",
            "CHECKPOINT_ROOT": (
                "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
                "state8_curriculum_repairs_20260812/"
                f"seed{seed}_{branch}/checkpoint-20"
            ),
            "CHECKPOINT_ADAPTER_SHA256": hashes["adapter_model"],
            "CHECKPOINT_STATE_SHA256": hashes["recurrent_state"],
            "CHECKPOINT_RECEIPT_SHA256": digest(receipt),
        }
        rendered_jobs.append(job)
    config["description"] = "Hash-bound causal-teacher cp20 dual diagnosis-C4 quickscreen"
    config["code"]["local_dir"] = str(PROJECT_ROOT)
    config["environment"]["setup"] = [
        "python -m venv --system-site-packages /tmp/studentmodeling-state8-repair-dual-c4-venv",
        "/tmp/studentmodeling-state8-repair-dual-c4-venv/bin/python -m pip install --upgrade pip",
        "/tmp/studentmodeling-state8-repair-dual-c4-venv/bin/python -m pip install -r requirements-feeds.txt",
    ]
    config["jobs"] = rendered_jobs
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.template, args.receipt_dir, args.output)
    print(f"materialized={args.output}")


if __name__ == "__main__":
    main()
