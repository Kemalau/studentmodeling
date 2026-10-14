#!/usr/bin/env python3
"""Receipt-last cache for deterministic State8 curriculum materializations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE = "state8_curriculum_data_cache_v1"
RECEIPT = "cache_receipt.json"
COMMON_FILES = (
    "common/valid.jsonl",
    "common/formal_eval.jsonl",
    "common/causal_same_rule/train.jsonl",
    "common/causal_same_rule/valid.jsonl",
    "common/causal_same_rule/augmentation_receipt.json",
    "common/causal_subset250/train.jsonl",
    "common/causal_subset250/valid.jsonl",
    "common/causal_subset250/train.jsonl.receipt.json",
)
VARIANT_FILES = (
    "train5000.jsonl",
    "train5000.jsonl.receipt.json",
)
VARIANTS = ("t3_heavy", "balanced")
PINNED_HASHES = {
    "common/valid.jsonl": "78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14",
    "common/formal_eval.jsonl": "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525",
    "common/causal_same_rule/train.jsonl": "e2a57273bd4c49229efc8cec522fc221e460174c388c43e2b0d4888cc5079225",
    "common/causal_same_rule/valid.jsonl": "8bd0dc3ec367ce30e5e5da3526681c21f852ba991394cac423682cd721f48c8e",
    "common/causal_subset250/valid.jsonl": "8bd0dc3ec367ce30e5e5da3526681c21f852ba991394cac423682cd721f48c8e",
    "t3_heavy/train5000.jsonl": "023e692afe932a6bbbd70547b698007c89b96ae7f7d79adae4622fdf34900540",
    "balanced/train5000.jsonl": "338ca9e62da39e03e8e2a25dec56c57179a4206ebf1dfef9729042b1ea681deb",
}
MANIFEST_HASHES = {
    "t3_heavy": "97bef6748b1fa8d75809e9e01c9b76d57b6534d336b7037217289932a31df393",
    "balanced": "baa0c9f7c062822cfffe11a5a4ec1e416bf726dfaf66fd8251f7dd6bb03bbadd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_files(variant: str | None = None) -> tuple[str, ...]:
    variants = (variant,) if variant else VARIANTS
    if any(value not in VARIANTS for value in variants):
        raise ValueError(f"Unsupported cache variant: {variant}")
    return COMMON_FILES + tuple(
        f"{value}/{name}" for value in variants for name in VARIANT_FILES
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_structural_receipts(root: Path, variant: str | None = None) -> None:
    variants = (variant,) if variant else VARIANTS
    for value in variants:
        subset = load_json(root / value / "train5000.jsonl.receipt.json")
        if subset.get("stage") != "materialized_recurrent_index_manifest_v1":
            raise ValueError(f"Subset receipt stage drift: {value}")
        if int(subset.get("rows", -1)) != 5000:
            raise ValueError(f"Subset row-count drift: {value}")
        if subset.get("manifest_sha256") != MANIFEST_HASHES[value]:
            raise ValueError(f"Subset manifest drift: {value}")
        if subset.get("output_sha256") != PINNED_HASHES[f"{value}/train5000.jsonl"]:
            raise ValueError(f"Subset output drift: {value}")
        if int(subset.get("formal_example_id_overlap", -1)) != 0:
            raise ValueError(f"Subset formal overlap: {value}")
    causal = load_json(root / "common/causal_same_rule/augmentation_receipt.json")
    if causal.get("stage") != "recurrent_causal_same_rule_augmentation_v1":
        raise ValueError("Causal augmentation receipt drift")
    if causal.get("status") != "passed":
        raise ValueError("Causal augmentation did not pass")
    twins = load_json(root / "common/causal_subset250/train.jsonl.receipt.json")
    if int(twins.get("pairs", -1)) != 250 or int(twins.get("twins", -1)) != 125:
        raise ValueError("Causal twin cardinality drift")
    if twins.get("role_counts") != {
        "recipient_native": 125,
        "same_rule_native": 125,
    }:
        raise ValueError("Causal twin role drift")


def create_receipt(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative in required_files():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Cache file missing: {path}")
        observed = sha256(path)
        expected = PINNED_HASHES.get(relative)
        if expected is not None and observed != expected:
            raise ValueError(
                f"Pinned cache hash drift: {relative} expected={expected} observed={observed}"
            )
        files[relative] = {"sha256": observed, "bytes": path.stat().st_size}
    validate_structural_receipts(root)
    receipt = {
        "stage": STAGE,
        "status": "complete",
        "receipt_published_last": True,
        "variants": list(VARIANTS),
        "pinned_hashes": PINNED_HASHES,
        "manifest_hashes": MANIFEST_HASHES,
        "files": files,
    }
    temporary = root / f"{RECEIPT}.partial"
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(root / RECEIPT)
    return receipt


def verify(root: Path, variant: str) -> dict[str, str]:
    receipt_path = root / RECEIPT
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Cache completion receipt missing: {receipt_path}")
    receipt = load_json(receipt_path)
    if receipt.get("stage") != STAGE or receipt.get("status") != "complete":
        raise ValueError("Cache receipt stage/status drift")
    if receipt.get("receipt_published_last") is not True:
        raise ValueError("Cache is not receipt-last")
    if receipt.get("pinned_hashes") != PINNED_HASHES:
        raise ValueError("Cache pinned-hash contract drift")
    if receipt.get("manifest_hashes") != MANIFEST_HASHES:
        raise ValueError("Cache manifest contract drift")
    file_receipts = receipt.get("files") or {}
    for relative in required_files(variant):
        metadata = file_receipts.get(relative) or {}
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Cache file missing: {path}")
        if int(metadata.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Cache size drift: {relative}")
        observed = sha256(path)
        if metadata.get("sha256") != observed:
            raise ValueError(f"Cache SHA256 drift: {relative}")
        expected = PINNED_HASHES.get(relative)
        if expected is not None and observed != expected:
            raise ValueError(f"Pinned cache SHA256 drift: {relative}")
    validate_structural_receipts(root, variant)
    return {
        "valid_jsonl": str(root / "common/valid.jsonl"),
        "formal_eval_jsonl": str(root / "common/formal_eval.jsonl"),
        "subset_jsonl": str(root / variant / "train5000.jsonl"),
        "causal_data": str(root / "common/causal_same_rule"),
        "causal_subset": str(root / "common/causal_subset250"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--cache-root", type=Path, required=True)
    check = subparsers.add_parser("verify")
    check.add_argument("--cache-root", type=Path, required=True)
    check.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    if args.command == "create":
        result: Any = create_receipt(args.cache_root)
    else:
        result = verify(args.cache_root, args.variant)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
