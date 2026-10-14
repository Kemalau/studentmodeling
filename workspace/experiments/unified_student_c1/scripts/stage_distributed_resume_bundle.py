#!/usr/bin/env python3
"""Stage and verify one atomic distributed checkpoint on node-local storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_bundle(root: Path) -> dict:
    state = root / "recurrent_state.pt"
    adapter_model = root / "adapter/adapter_model.safetensors"
    adapter_config = root / "adapter/adapter_config.json"
    receipt_path = root / "checkpoint_completion.json"
    for path in (state, adapter_model, adapter_config, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Resume bundle artifact is missing: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise ValueError(f"Resume bundle receipt is incomplete: {receipt}")
    expected = receipt.get("sha256")
    actual = {
        "adapter_model": sha256(adapter_model),
        "recurrent_state": sha256(state),
    }
    if expected != actual:
        raise ValueError(
            f"Resume bundle hash mismatch: expected={expected} actual={actual}"
        )
    return receipt


def stage_bundle(
    source_state: Path,
    destination: Path,
    *,
    expected_state_sha256: str | None = None,
) -> dict:
    source_state = source_state.resolve()
    source_root = source_state.parent
    source_adapter_model = source_root / "adapter/adapter_model.safetensors"
    source_adapter_config = source_root / "adapter/adapter_config.json"
    source_receipt_path = source_root / "checkpoint_completion.json"
    for path in (
        source_state,
        source_adapter_model,
        source_adapter_config,
        source_receipt_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Source resume bundle artifact is missing: {path}")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if source_receipt.get("status") != "complete":
        raise ValueError(f"Source resume bundle receipt is incomplete: {source_receipt}")
    if not isinstance(source_receipt.get("sha256"), dict):
        raise ValueError("Source resume bundle receipt lacks artifact hashes")
    if (
        expected_state_sha256 is not None
        and source_receipt["sha256"]["recurrent_state"] != expected_state_sha256
    ):
        raise ValueError("Authoritative source checkpoint state SHA256 drift")

    destination.mkdir(parents=True, exist_ok=True)
    source_marker = destination / "recurrent_state.source"
    complete_marker = destination / "recurrent_state.complete"
    if complete_marker.is_file() and source_marker.is_file():
        if source_marker.read_text(encoding="utf-8").strip() == str(source_state):
            try:
                return validate_bundle(destination)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                pass

    state = destination / "recurrent_state.pt"
    receipt = destination / "checkpoint_completion.json"
    adapter = destination / "adapter"
    state_partial = destination / "recurrent_state.pt.partial"
    receipt_partial = destination / "checkpoint_completion.json.partial"
    adapter_partial = destination / "adapter.partial"
    marker_partial = destination / "recurrent_state.source.partial"

    complete_marker.unlink(missing_ok=True)
    for path in (state_partial, receipt_partial, marker_partial):
        path.unlink(missing_ok=True)
    shutil.rmtree(adapter_partial, ignore_errors=True)

    shutil.copyfile(source_state, state_partial)
    shutil.copytree(source_root / "adapter", adapter_partial)
    shutil.copyfile(source_receipt_path, receipt_partial)
    staged_receipt = json.loads(receipt_partial.read_text(encoding="utf-8"))
    staged_actual = {
        "adapter_model": sha256(adapter_partial / "adapter_model.safetensors"),
        "recurrent_state": sha256(state_partial),
    }
    if staged_receipt.get("status") != "complete" or (
        staged_receipt.get("sha256") != staged_actual
    ):
        raise ValueError("Node-local staged resume bundle failed hash verification")

    state.unlink(missing_ok=True)
    receipt.unlink(missing_ok=True)
    shutil.rmtree(adapter, ignore_errors=True)
    os.replace(state_partial, state)
    os.replace(adapter_partial, adapter)
    marker_partial.write_text(str(source_state) + "\n", encoding="utf-8")
    os.replace(marker_partial, source_marker)
    # The authoritative completion receipt is published last.
    os.replace(receipt_partial, receipt)
    validate_bundle(destination)
    complete_marker.touch()
    return staged_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-state-sha256")
    args = parser.parse_args()
    receipt = stage_bundle(
        args.source_state,
        args.destination,
        expected_state_sha256=args.expected_state_sha256,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
