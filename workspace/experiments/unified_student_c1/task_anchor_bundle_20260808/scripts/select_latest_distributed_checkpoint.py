#!/usr/bin/env python3
"""Select the newest complete, hash-verified distributed checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_round(checkpoint: Path) -> int | None:
    try:
        round_index = int(checkpoint.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None
    receipt_path = checkpoint / "checkpoint_completion.json"
    state_path = checkpoint / "recurrent_state.pt"
    adapter_path = checkpoint / "adapter/adapter_model.safetensors"
    if not (receipt_path.is_file() and state_path.is_file() and adapter_path.is_file()):
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    actual = {
        "adapter_model": sha256(adapter_path),
        "recurrent_state": sha256(state_path),
    }
    if (
        receipt.get("status") != "complete"
        or int(receipt.get("round", -1)) != round_index
        or actual != (receipt.get("sha256") or {})
    ):
        return None
    return round_index


def select_latest(root: Path, *, max_round: int | None = None) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not root.is_dir():
        return None
    for checkpoint in root.glob("checkpoint-round-*"):
        round_index = verified_round(checkpoint)
        if round_index is None:
            continue
        if max_round is not None and round_index > max_round:
            continue
        candidates.append((round_index, checkpoint / "recurrent_state.pt"))
    return max(candidates)[1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-round", type=int)
    args = parser.parse_args()
    selected = select_latest(args.root, max_round=args.max_round)
    if selected is not None:
        print(selected)


if __name__ == "__main__":
    main()
