#!/usr/bin/env python3
"""Bind the frozen unified screen to one immutable repair checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

try:
    from audit_state8_curriculum_repair_checkpoint import SPECS, STEPS
except ImportError:  # package-style unit tests
    from scripts.audit_state8_curriculum_repair_checkpoint import SPECS, STEPS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    template: Path,
    receipt_path: Path,
    seed: int,
    branch: str,
    step: int,
    output: Path,
) -> None:
    if (seed, branch) not in SPECS or step not in STEPS:
        raise ValueError("Unsupported repair checkpoint coordinate")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or int(receipt.get("step", -1)) != step:
        raise ValueError("Repair checkpoint completion receipt is incomplete")
    hashes = receipt.get("sha256") or {}
    if set(hashes) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Repair checkpoint artifact hash schema drift")
    values = (*hashes.values(), digest(receipt_path))
    if any(
        len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in values
    ):
        raise ValueError("Invalid repair checkpoint SHA256")
    root = (
        "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
        "state8_curriculum_repairs_20260812/"
        f"seed{seed}_{branch}/checkpoint-{step}"
    )
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["description"] = "State8 curriculum repair checkpoint unified screen"
    config["code"]["local_dir"] = str(PROJECT_ROOT)
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Template job matrix drift")
    job = jobs[0]
    arm = f"repair_{branch}_seed{seed}"
    job["name"] = (
        f"state8-repair-{branch.replace('_', '-')}-seed{seed}-cp{step}-screen"
    )
    runner = Path(__file__).with_name("run_state8_curriculum_checkpoint_screen.sh")
    job["command"] = str(job["command"]).replace("RUNNER_SHA256", digest(runner))
    job["submit_args"]["env"].update(
        CHECKPOINT_FAMILY="repair",
        CURRICULUM_ARM=arm,
        REPAIR_SEED=str(seed),
        REPAIR_BRANCH=branch,
        CHECKPOINT_STEP=str(step),
        CHECKPOINT_ROOT=root,
        CHECKPOINT_ADAPTER_SHA256=hashes["adapter_model"],
        CHECKPOINT_STATE_SHA256=hashes["recurrent_state"],
        CHECKPOINT_RECEIPT_SHA256=values[2],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(
        args.template,
        args.receipt,
        args.seed,
        args.branch,
        args.step,
        args.output,
    )
    print(f"materialized={args.output}")


if __name__ == "__main__":
    main()
