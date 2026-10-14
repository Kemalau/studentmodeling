#!/usr/bin/env python3
"""Fail closed on the optimizer-boundary payload used by State8 recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit(checkpoint: Path, completion_path: Path) -> dict[str, Any]:
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(completion, dict):
        raise ValueError("Checkpoint or completion receipt is not an object")
    step = int(payload.get("step", -1))
    if step != 2000 or step % 4:
        raise ValueError(f"Checkpoint is not the approved GA=4 boundary: {step}")
    config = payload.get("config") or {}
    required = (
        "optimizer",
        "scheduler",
        "python_random_state",
        "torch_random_state",
        "cuda_random_state_all",
        "real_order",
        "causal_schedule",
        "history",
        "transition_in",
        "transition_out",
    )
    missing = [name for name in required if payload.get(name) is None]
    if missing:
        raise ValueError(f"Resume payload is incomplete: {missing}")
    if int(config.get("gradient_accumulation_steps", -1)) != 4:
        raise ValueError("Gradient-accumulation contract drift")
    if int(config.get("planned_optimizer_steps", -1)) != 1250:
        raise ValueError("Optimizer-step contract drift")
    if int(config.get("full_micro_steps", -1)) != 5000:
        raise ValueError("Five-stage curriculum length drift")
    if config.get("history_dropout_schedule") != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("History-dropout curriculum drift")
    observed = digest(checkpoint)
    if completion.get("status") != "complete" or int(completion.get("step", -1)) != step:
        raise ValueError("Checkpoint completion receipt drift")
    if (completion.get("sha256") or {}).get("recurrent_state") != observed:
        raise ValueError("Checkpoint hash disagrees with completion receipt")
    return {
        "stage": "state8_single_resume_checkpoint_audit_v1",
        "status": "complete",
        "step": step,
        "optimizer_step": step // 4,
        "remaining_micro_steps": 5000 - step,
        "recurrent_state_sha256": observed,
        "required_payload_keys": list(required),
        "history_dropout_schedule": config["history_dropout_schedule"],
        "causal_objective_mode": config.get("causal_objective_mode"),
        "causal_repeat_fraction": config.get("causal_repeat_fraction"),
        "state_token_count": config.get("state_token_count"),
        "final_inference_route": config.get("final_inference_route"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.completion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
