#!/usr/bin/env python3
"""Bind a same-question checkpoint screen to an immutable completion receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ARMS = {
    "updater": {
        "audit_arm": "eedi_same_question_updater_mse1",
        "seed": 3072,
        "tag": "source_anchor_updater_mse1_eedi_same_question_v1",
    },
    "full": {
        "audit_arm": "eedi_same_question_full_mse1",
        "seed": 3071,
        "tag": "source_anchor_full_mse1_eedi_same_question_v1",
    },
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    template: Path, receipt_path: Path, arm: str, step: int, output: Path
) -> None:
    if arm not in ARMS or step not in {20, 100}:
        raise ValueError("Unsupported arm/checkpoint step")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if set(receipt) != {"status", "step", "sha256"} or receipt.get("status") != "complete":
        raise ValueError("Checkpoint completion receipt is incomplete")
    if int(receipt.get("step", -1)) != step:
        raise ValueError("Checkpoint step drift")
    hashes = receipt.get("sha256") or {}
    if set(hashes) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Checkpoint artifact hash schema drift")
    values = (*hashes.values(), digest(receipt_path))
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in values):
        raise ValueError("Invalid checkpoint SHA256")
    spec = ARMS[arm]
    root = (
        "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
        f"state8_eedi_same_question_20260811/{arm}/checkpoint-{step}"
    )
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["code"]["local_dir"] = str(PROJECT_ROOT)
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Template job matrix drift")
    job = jobs[0]
    job["name"] = f"state8-eedi-same-question-{arm}-cp{step}-screen"
    runner = Path(__file__).with_name("run_state8_eedi_same_question_checkpoint_screen.sh")
    job["command"] = str(job["command"]).replace("RUNNER_SHA256", digest(runner))
    env = job["submit_args"]["env"]
    env.update(
        CHECKPOINT_STEP=str(step), CHECKPOINT_ARM=spec["audit_arm"], CHECKPOINT_ROOT=root,
        CHECKPOINT_ADAPTER_SHA256=hashes["adapter_model"],
        CHECKPOINT_STATE_SHA256=hashes["recurrent_state"],
        CHECKPOINT_RECEIPT_SHA256=values[2],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


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
