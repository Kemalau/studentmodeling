#!/usr/bin/env python3
"""Quarantine receipts that prove an incompatible distributed recipe."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch


ROUND = re.compile(r"^checkpoint-round-([1-9][0-9]*)$")


def planned_steps(state: Path) -> int | None:
    payload = torch.load(state, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        return None
    value = payload["config"].get("planned_optimizer_steps")
    return value if type(value) is int and value > 0 else None


def quarantine(run_dir: Path, expected: int) -> dict[str, Any]:
    if expected <= 0:
        raise ValueError("expected planned optimizer steps must be positive")
    changed: list[dict[str, Any]] = []
    inspected = 0
    if not run_dir.is_dir():
        return {"stage": "incompatible_checkpoint_receipt_quarantine_v1",
                "expected_planned_optimizer_steps": expected,
                "run_dir": str(run_dir), "inspected": 0, "quarantined": []}
    for checkpoint in sorted(run_dir.glob("checkpoint-round-*")):
        match = ROUND.fullmatch(checkpoint.name)
        receipt = checkpoint / "checkpoint_completion.json"
        state = checkpoint / "recurrent_state.pt"
        if match is None or not receipt.is_file() or not state.is_file():
            continue
        inspected += 1
        try:
            observed = planned_steps(state)
        except (OSError, RuntimeError, ValueError, TypeError):
            continue
        if observed is None or observed == expected:
            continue
        destination = checkpoint / f"checkpoint_completion.incompatible-planned-{observed}.json"
        original = receipt.read_bytes()
        if destination.exists() and destination.read_bytes() != original:
            raise ValueError(f"Conflicting quarantine receipt: {destination}")
        if destination.exists():
            receipt.unlink()
        else:
            os.replace(receipt, destination)
        changed.append({"checkpoint": checkpoint.name, "round": int(match.group(1)),
                        "observed_planned_optimizer_steps": observed,
                        "quarantined_receipt": destination.name})
    return {"stage": "incompatible_checkpoint_receipt_quarantine_v1",
            "expected_planned_optimizer_steps": expected, "run_dir": str(run_dir),
            "inspected": inspected, "quarantined": changed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-planned-optimizer-steps", type=int, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = quarantine(args.run_dir, args.expected_planned_optimizer_steps)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.audit.with_suffix(args.audit.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.audit)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
