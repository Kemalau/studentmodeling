#!/usr/bin/env python3
"""Fail closed if the frozen XES profile holdout or protocol has drifted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from build_dbe_student_profile_pilot import sha256
    from evaluate_dbe_student_profile_transport_v3 import validate_rows
except ModuleNotFoundError:
    from scripts.build_dbe_student_profile_pilot import sha256
    from scripts.evaluate_dbe_student_profile_transport_v3 import validate_rows


EXPECTED = {
    "train": (800, "0df123f919eeb00b8aa1902ba8fa72f67f68a5b254e9c799cf93a9a03a06a4c0"),
    "valid": (200, "72947f89e7d4ba5571120e0c7c90c3f723ad3f37d176061e7b77accff3610145"),
    "test": (466, "aa4f08f11453372d32229de2cb4fa7547f1de90379d6feffcfbe58de1c7fb1a2"),
}
EXPECTED_MANIFEST = "917f81d4d0bb9e8768efaa3e11c5993184979bdc68143e7f9c10bd863606b769"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if sha256(manifest_path) != EXPECTED_MANIFEST:
        raise ValueError("Confirmatory manifest hash drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_confirmatory_evaluation":
        raise ValueError("Confirmatory protocol is not frozen")
    if manifest.get("official_test_metrics_inspected_before_freeze") is not False:
        raise ValueError("Confirmatory inspection declaration drift")
    rows = {}
    for split, (expected_n, expected_hash) in EXPECTED.items():
        path = root / f"{split}.jsonl"
        if sha256(path) != expected_hash:
            raise ValueError(f"Frozen {split} JSONL hash drift")
        rows[split] = read_jsonl(path)
        if len(rows[split]) != expected_n:
            raise ValueError(f"Frozen {split} row cardinality drift")
    validate_rows(rows)
    sets = {
        split: {str(row["student_id"]) for row in values}
        for split, values in rows.items()
    }
    if any(
        sets[left] & sets[right]
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    ):
        raise ValueError("Frozen confirmatory students overlap")
    if any(
        not bool(row["target_provenance"].get("official_test_student"))
        for row in rows["test"]
    ):
        raise ValueError("A confirmatory row is not from the official test split")
    return {
        "status": "pass",
        "manifest_sha256": EXPECTED_MANIFEST,
        "rows": {split: len(values) for split, values in rows.items()},
        "test_sha256": EXPECTED["test"][1],
        "student_disjoint": True,
        "test_metrics_inspected_by_this_verifier": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("manifests/xes_state8_profile_confirmatory_v1"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
