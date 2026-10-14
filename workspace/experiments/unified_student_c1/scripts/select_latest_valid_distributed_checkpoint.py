#!/usr/bin/env python3
"""Select the newest hash-valid atomic distributed checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint_round(path: Path) -> int | None:
    prefix = "checkpoint-round-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    try:
        value = int(path.name[len(prefix) :])
    except ValueError:
        return None
    return value if value > 0 else None


def validate_candidate(
    path: Path,
    *,
    expected_world_size: int,
    max_round: int,
    save_steps: int,
    recovery_save_steps: int = 0,
    expected_planned_optimizer_steps: int | None = None,
) -> tuple[bool, str]:
    round_index = checkpoint_round(path)
    if round_index is None:
        return False, "invalid checkpoint directory name"
    if round_index > max_round:
        return False, "checkpoint round exceeds the exact epoch"
    normal_cadence = round_index % save_steps == 0
    recovery_cadence = (
        recovery_save_steps > 0 and round_index % recovery_save_steps == 0
    )
    if round_index != max_round and not (normal_cadence or recovery_cadence):
        return False, "checkpoint round is outside the frozen save cadence"
    receipt_path = path / "checkpoint_completion.json"
    adapter = path / "adapter/adapter_model.safetensors"
    state = path / "recurrent_state.pt"
    for required in (receipt_path, adapter, state):
        if not required.is_file():
            return False, f"missing {required.relative_to(path)}"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid completion receipt: {error}"
    if receipt.get("status") != "complete":
        return False, "completion status is not complete"
    if receipt.get("round") != round_index:
        return False, "receipt round does not match directory"
    if receipt.get("world_size") != expected_world_size:
        return False, "receipt world size drift"
    expected = receipt.get("sha256")
    if not isinstance(expected, dict):
        return False, "receipt hashes are missing"
    try:
        actual = {
            "adapter_model": sha256(adapter),
            "recurrent_state": sha256(state),
        }
    except OSError as error:
        return False, f"artifact read failed: {error}"
    if actual != expected:
        return False, "artifact hash mismatch"
    if expected_planned_optimizer_steps is not None:
        try:
            payload = torch.load(state, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            return False, f"checkpoint payload is unreadable: {error}"
        config = payload.get("config") if isinstance(payload, dict) else None
        if not isinstance(config, dict):
            return False, "checkpoint payload lacks training config"
        observed_steps = config.get("planned_optimizer_steps")
        if observed_steps != expected_planned_optimizer_steps:
            return False, (
                "planned optimizer steps drift: "
                f"checkpoint={observed_steps!r} expected={expected_planned_optimizer_steps}"
            )
    return True, "valid"


def select(
    run_dir: Path,
    *,
    expected_world_size: int = 16,
    max_round: int = 16_241,
    save_steps: int = 250,
    recovery_save_steps: int = 0,
    probe_expected_cadence: bool = False,
    expected_planned_optimizer_steps: int | None = None,
) -> Path | None:
    if expected_world_size <= 1:
        raise ValueError("expected_world_size must be greater than one")
    if max_round <= 0 or save_steps <= 0 or recovery_save_steps < 0:
        raise ValueError(
            "max_round/save_steps must be positive and recovery_save_steps non-negative"
        )
    if (
        expected_planned_optimizer_steps is not None
        and expected_planned_optimizer_steps <= 0
    ):
        raise ValueError("expected_planned_optimizer_steps must be positive")
    if not run_dir.is_dir():
        return None
    candidates: dict[int, Path] = {}
    for path in run_dir.glob("checkpoint-round-*"):
        round_index = checkpoint_round(path)
        if round_index is not None:
            candidates[round_index] = path
    if probe_expected_cadence:
        # Blob/FUSE directory listings can lag behind direct path reads on a
        # freshly scheduled node. Probe the frozen save cadences explicitly so
        # a complete remote receipt cannot be missed merely because glob() is
        # stale. validate_candidate remains the authority for hashes, world
        # size, bounds, and completion status.
        cadences = {save_steps}
        if recovery_save_steps > 0:
            cadences.add(recovery_save_steps)
        rounds = {max_round}
        for cadence in cadences:
            rounds.update(range(cadence, max_round + 1, cadence))
        for round_index in rounds:
            path = run_dir / f"checkpoint-round-{round_index}"
            if (path / "checkpoint_completion.json").is_file():
                candidates[round_index] = path
    for _, path in sorted(candidates.items(), reverse=True):
        valid, reason = validate_candidate(
            path,
            expected_world_size=expected_world_size,
            max_round=max_round,
            save_steps=save_steps,
            recovery_save_steps=recovery_save_steps,
            expected_planned_optimizer_steps=expected_planned_optimizer_steps,
        )
        if valid:
            return path / "recurrent_state.pt"
        print(f"skipping invalid checkpoint {path.name}: {reason}", file=sys.stderr)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=16)
    parser.add_argument("--max-round", type=int, default=16_241)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--recovery-save-steps", type=int, default=0)
    parser.add_argument("--expected-planned-optimizer-steps", type=int)
    parser.add_argument(
        "--probe-expected-cadence",
        action="store_true",
        help=(
            "Directly probe frozen checkpoint paths in addition to globbing; "
            "use this on eventually consistent Blob/FUSE mounts."
        ),
    )
    args = parser.parse_args()
    selected = select(
        args.run_dir,
        expected_world_size=args.expected_world_size,
        max_round=args.max_round,
        save_steps=args.save_steps,
        recovery_save_steps=args.recovery_save_steps,
        probe_expected_cadence=args.probe_expected_cadence,
        expected_planned_optimizer_steps=args.expected_planned_optimizer_steps,
    )
    if selected is not None:
        print(selected)


if __name__ == "__main__":
    main()
