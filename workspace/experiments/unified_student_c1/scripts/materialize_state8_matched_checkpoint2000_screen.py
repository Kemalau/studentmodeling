#!/usr/bin/env python3
"""Bind the checkpoint-2000 screen template to its atomic receipt hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def materialize(template: Path, receipt: Path, output: Path, step: int = 2000) -> dict:
    value = json.loads(receipt.read_text(encoding="utf-8"))
    expected_keys = {"adapter_model", "recurrent_state"}
    if step not in {2000, 3000, 4000, 5000}:
        raise ValueError("Unsupported matched checkpoint step")
    if value.get("status") != "complete" or value.get("step") != step or set(value.get("sha256") or {}) != expected_keys:
        raise ValueError(f"Checkpoint-{step} receipt drift")
    adapter = value["sha256"]["adapter_model"]; state = value["sha256"]["recurrent_state"]
    for digest in (adapter, state):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Invalid artifact digest")
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    text = template.read_text(encoding="utf-8")
    replacements = {"HASH_ADAPTER_SHA256": adapter, "HASH_STATE_SHA256": state, "HASH_RECEIPT_SHA256": receipt_sha}
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"Template placeholder drift: {old}")
        text = text.replace(old, new)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(text, encoding="utf-8"); os.replace(partial, output)
    return {"adapter_model": adapter, "recurrent_state": state, "receipt": receipt_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=2000)
    args = parser.parse_args(); print(json.dumps(materialize(args.template, args.receipt, args.output, args.step), sort_keys=True))


if __name__ == "__main__":
    main()
