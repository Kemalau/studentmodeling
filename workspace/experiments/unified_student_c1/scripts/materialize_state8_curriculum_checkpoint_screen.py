#!/usr/bin/env python3
"""Bind the unified-task curriculum screen to one immutable checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ARMS = {
    "generic": "state8_diagnosis_first_eedi_curriculum_20260812",
    "current_query": "state8_diagnosis_first_eedi_current_query_curriculum_20260812",
}
STEPS = {2000, 3000, 4000, 5000}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    template: Path,
    receipt_path: Path,
    arm: str,
    step: int,
    output: Path,
) -> None:
    if arm not in ARMS or step not in STEPS:
        raise ValueError("Unsupported curriculum arm/checkpoint step")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    position = receipt.get("step", receipt.get("round"))
    if receipt.get("status") != "complete" or int(position or -1) != step:
        raise ValueError("Checkpoint completion receipt is incomplete")
    hashes = receipt.get("sha256") or {}
    if set(hashes) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Checkpoint artifact hash schema drift")
    values = (*hashes.values(), digest(receipt_path))
    if any(
        len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in values
    ):
        raise ValueError("Invalid checkpoint SHA256")
    root = (
        "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
        f"{ARMS[arm]}/checkpoint-{step}"
    )
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["code"]["local_dir"] = str(PROJECT_ROOT)
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Template job matrix drift")
    job = jobs[0]
    job["name"] = f"state8-curriculum-{arm.replace('_', '-')}-cp{step}-screen"
    runner = Path(__file__).with_name("run_state8_curriculum_checkpoint_screen.sh")
    job["command"] = str(job["command"]).replace("RUNNER_SHA256", digest(runner))
    job["submit_args"]["env"].update(
        CHECKPOINT_FAMILY="curriculum",
        CURRICULUM_ARM=arm,
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
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.template, args.receipt, args.arm, args.step, args.output)
    print(f"materialized={args.output}")


if __name__ == "__main__":
    main()
