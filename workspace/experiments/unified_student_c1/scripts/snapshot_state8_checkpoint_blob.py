#!/usr/bin/env python3
"""Create a receipt-last same-account Blob snapshot of one State8 checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ACCOUNT = "csnewsandfeeds4150361735"
CONTAINER = "amulet"
FILES = (
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "recurrent_state.pt",
)


def az(*arguments: str) -> str:
    result = subprocess.run(
        ["az", "storage", "blob", *arguments, "--account-name", ACCOUNT,
         "--auth-mode", "login", "--only-show-errors", "-o", "json"],
        text=True, capture_output=True, check=True,
    )
    return result.stdout


def exists(name: str) -> bool:
    value = json.loads(az("exists", "--container-name", CONTAINER, "--name", name))
    return bool(value.get("exists"))


def copy_status(destination: str) -> tuple[str, str]:
    value = json.loads(az(
        "show", "--container-name", CONTAINER, "--name", destination,
    ))
    properties = value.get("properties") or {}
    copy = properties.get("copy") or value.get("copy") or {}
    status = str(copy.get("status") or value.get("copy_status") or "").casefold()
    description = str(
        copy.get("statusDescription")
        or copy.get("status_description")
        or value.get("copy_status_description")
        or ""
    )
    return status, description


def copy(source: str, destination: str, *, timeout_seconds: int = 3600) -> None:
    # Synchronous Blob copy is capped at 256 MiB.  State8 adapters can exceed
    # that limit, so use server-side asynchronous copy and do not publish the
    # completion receipt until Azure reports success for every artifact.
    value = json.loads(az(
        "copy", "start", "--destination-container", CONTAINER,
        "--destination-blob", destination, "--source-account-name", ACCOUNT,
        "--source-container", CONTAINER, "--source-blob", source,
    ))
    status = str(value.get("copy_status") or "").casefold()
    if status == "success":
        return
    if status not in {"", "pending"}:
        raise RuntimeError(f"Checkpoint snapshot copy failed to start: {value}")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status, description = copy_status(destination)
        if status == "success":
            return
        if status in {"failed", "aborted"}:
            raise RuntimeError(
                f"Checkpoint snapshot copy {status}: {description or destination}"
            )
        if status not in {"", "pending"}:
            raise RuntimeError(
                f"Checkpoint snapshot copy returned unknown status={status!r}: "
                f"{description or destination}"
            )
        time.sleep(2)
    raise TimeoutError(f"Checkpoint snapshot copy timed out: {destination}")


def snapshot(source: str, destination: str, receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or set(receipt.get("sha256") or {}) != {
        "adapter_model", "recurrent_state"
    }:
        raise ValueError("Checkpoint completion receipt is incomplete")
    if exists(f"{destination}/checkpoint_completion.json"):
        return {"status":"exists","source":source,"destination":destination,"receipt":receipt}
    position = receipt.get("step", receipt.get("round"))
    if not isinstance(position, int) or position <= 0:
        raise ValueError("Checkpoint completion receipt lacks a positive step/round")
    for relative in FILES:
        source_blob=f"{source}/{relative}"
        if not exists(source_blob):
            raise FileNotFoundError(source_blob)
        copy(source_blob, f"{destination}/{relative}")
    copy(f"{source}/checkpoint_completion.json", f"{destination}/checkpoint_completion.json")
    return {"status":"complete","source":source,"destination":destination,
            "receipt_last":True,"files":[*FILES,"checkpoint_completion.json"],
            "artifact_sha256":receipt["sha256"],"step":position}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--destination-prefix", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    result=snapshot(args.source_prefix,args.destination_prefix,args.receipt)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
