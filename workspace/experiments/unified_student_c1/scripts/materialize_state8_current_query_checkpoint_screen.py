#!/usr/bin/env python3
"""Bind one query-aligned milestone screen to its checkpoint receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(template: Path, receipt_path: Path, step: int, output: Path) -> None:
    if step not in {20, 100}:
        raise ValueError("Only checkpoint 20 and 100 screens are supported")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_keys = {"status", "step", "sha256"}
    if set(receipt) != expected_keys or receipt.get("status") != "complete":
        raise ValueError("Checkpoint completion receipt is incomplete")
    if int(receipt.get("step", -1)) != step:
        raise ValueError("Checkpoint step drift")
    hashes = receipt.get("sha256") or {}
    if set(hashes) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Checkpoint artifact hash schema drift")
    values = (*hashes.values(), digest(receipt_path))
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in values
    ):
        raise ValueError("Invalid checkpoint SHA256")
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["code"]["local_dir"] = str(PROJECT_ROOT)
    job_name = f"state8-current-query-cp{step}-screen"
    selected = [job for job in config.get("jobs") or [] if job.get("name") == job_name]
    if len(selected) != 1:
        raise ValueError("Template job matrix drift")
    job = selected[0]
    env = job["submit_args"]["env"]
    env.update(
        CHECKPOINT_STEP=str(step),
        CHECKPOINT_ADAPTER_SHA256=hashes["adapter_model"],
        CHECKPOINT_STATE_SHA256=hashes["recurrent_state"],
        CHECKPOINT_RECEIPT_SHA256=values[2],
    )
    config["jobs"] = [job]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.template, args.receipt, args.step, args.output)
    print(f"materialized={args.output}")


if __name__ == "__main__":
    main()
