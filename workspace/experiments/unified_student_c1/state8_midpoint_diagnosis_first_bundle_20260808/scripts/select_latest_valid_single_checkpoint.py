#!/usr/bin/env python3
"""Select the newest hash-valid atomic single-device checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int | None:
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    if suffix.startswith("round-"):
        return None
    try:
        step = int(suffix)
    except ValueError:
        return None
    return step if step > 0 else None


def validate_candidate(
    path: Path,
    *,
    max_step: int,
    save_steps: int,
    planned_optimizer_steps: int,
    gradient_accumulation_steps: int,
) -> tuple[bool, str]:
    step = checkpoint_step(path)
    if step is None or step > max_step:
        return False, "invalid or out-of-range checkpoint step"
    if step % gradient_accumulation_steps:
        return False, "checkpoint is not on a completed gradient-accumulation boundary"
    if step != max_step and step % save_steps:
        return False, "checkpoint is outside the frozen save cadence"
    receipt_path = path / "checkpoint_completion.json"
    adapter = path / "adapter/adapter_model.safetensors"
    state = path / "recurrent_state.pt"
    if any(not item.is_file() for item in (receipt_path, adapter, state)):
        return False, "checkpoint bundle is incomplete"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid completion receipt: {error}"
    expected_hashes = receipt.get("sha256")
    if receipt.get("status") != "complete" or receipt.get("step") != step:
        return False, "completion receipt identity drift"
    try:
        actual_hashes = {
            "adapter_model": sha256(adapter),
            "recurrent_state": sha256(state),
        }
    except OSError as error:
        return False, f"artifact read failed: {error}"
    if expected_hashes != actual_hashes:
        return False, "artifact hash mismatch"
    try:
        payload = torch.load(state, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return False, f"checkpoint payload is unreadable: {error}"
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        return False, "checkpoint payload lacks training config"
    expected = {
        "distributed": False,
        "distributed_world_size": 1,
        "planned_optimizer_steps": planned_optimizer_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "full_micro_steps": max_step,
    }
    drift = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if drift:
        return False, f"single-device recipe drift: {drift}"
    if int(payload.get("step", -1)) != step:
        return False, "payload step drift"
    history = payload.get("history") or []
    if len(history) != step:
        return False, "checkpoint history length drift"
    return True, "valid"


def select(
    run_dir: Path,
    *,
    max_step: int,
    save_steps: int,
    planned_optimizer_steps: int,
    gradient_accumulation_steps: int,
) -> Path | None:
    if min(max_step, save_steps, planned_optimizer_steps, gradient_accumulation_steps) <= 0:
        raise ValueError("checkpoint bounds must be positive")
    if not run_dir.is_dir():
        return None
    candidates: dict[int, Path] = {}
    for step in range(save_steps, max_step + 1, save_steps):
        path = run_dir / f"checkpoint-{step}"
        if (path / "checkpoint_completion.json").is_file():
            candidates[step] = path
    final = run_dir / f"checkpoint-{max_step}"
    if (final / "checkpoint_completion.json").is_file():
        candidates[max_step] = final
    for _, path in sorted(candidates.items(), reverse=True):
        valid, reason = validate_candidate(
            path,
            max_step=max_step,
            save_steps=save_steps,
            planned_optimizer_steps=planned_optimizer_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        if valid:
            return path / "recurrent_state.pt"
        print(f"skipping invalid checkpoint {path.name}: {reason}", file=sys.stderr)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--planned-optimizer-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    args = parser.parse_args()
    selected = select(
        args.run_dir,
        max_step=args.max_step,
        save_steps=args.save_steps,
        planned_optimizer_steps=args.planned_optimizer_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if selected is not None:
        print(selected)


if __name__ == "__main__":
    main()
