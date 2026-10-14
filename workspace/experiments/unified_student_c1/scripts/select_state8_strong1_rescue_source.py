#!/usr/bin/env python3
"""Choose the closest hash-bound failed bridge candidate for rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREFIX = (
    "/mnt/amulet/studentmodeling/diagnostics/"
    "state8_qualified_teacher_curriculum_highkd_v1_immutable_"
    "taxonomy_semantic_v1_strong1_widen8_"
)
SLUGS = {"lr5e7_seed58", "lr1e6_seed59", "single_lr5e7_seed60"}


def remote_root(slug: str) -> str:
    suffix = slug + ("_single_ga4" if slug == "single_lr5e7_seed60" else "")
    return PREFIX + suffix


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def penalty(candidate: dict[str, Any]) -> float:
    checks = candidate.get("checks") or {}
    floors = candidate.get("floors") or {}
    value = 0.0
    for key in ("answer_C4_pass", "diagnosis_C4_pass"):
        value += 1.0 if checks.get(key) is not True else 0.0
    for key in (
        "T1", "T2", "T3_eedi_semantic", "T3_synthetic_semantic",
        "T3_schema", "T5", "T6", "T6_macro_f1",
        "answer_C4_D_specific", "diagnosis_C4_D_specific",
        "diagnosis_C4_swap_to_donor", "diagnosis_C4_same_rule_retention",
        "diagnosis_C4_noop_retention",
    ):
        floor = float(floors[key])
        observed = float(checks[key])
        value += max(0.0, floor - observed) / max(abs(floor), 1e-6)
    if float(checks.get("T6_positive_recall", 0.0)) <= 0.0:
        value += 1.0
    for key, floor_key in (
        ("answer_C4_same_rule", "answer_C4_same_rule_max"),
        ("answer_C4_noop", "answer_C4_noop_max"),
        ("diagnosis_C4_random_stable", "diagnosis_C4_random_stable_max"),
    ):
        maximum = float(floors[floor_key])
        observed = float(checks[key])
        value += max(0.0, observed - maximum) / max(abs(maximum), 1e-6)
    return value


def select(paths: list[Path]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        value = read(path)
        if value.get("stage") != "state8_strong1_widen8_joint_promotion_v1":
            raise ValueError(f"Selection stage drift: {path}")
        if value.get("pass") is True:
            raise ValueError("Rescue source selection is forbidden after a qualified bridge")
        for row in value.get("candidates") or []:
            slug = str(row.get("slug") or "")
            if slug not in SLUGS or row.get("qualified") is True:
                raise ValueError(f"Candidate drift: {slug}")
            artifacts = row.get("artifact_sha256") or {}
            if set(artifacts) != {"adapter_model", "recurrent_state"} or any(
                not isinstance(digest, str) or len(digest) != 64
                for digest in artifacts.values()
            ):
                raise ValueError(f"Artifact identity drift: {slug}")
            observed_root = str(row.get("remote_root") or "")
            if observed_root != remote_root(slug):
                raise ValueError(f"Remote root drift: {slug}: {observed_root}")
            candidates.append({**row, "source_selection": str(path), "penalty": penalty(row)})
    if not candidates:
        raise ValueError("No failed bridge candidates are available for rescue")
    by_slug: dict[str, dict[str, Any]] = {}
    for row in candidates:
        slug = str(row["slug"])
        if slug in by_slug:
            raise ValueError(f"Duplicate rescue candidate: {slug}")
        by_slug[slug] = row
    selected = min(candidates, key=lambda row: (float(row["penalty"]), -float(row["score"]), str(row["slug"])))
    return {
        "stage": "state8_strong1_rescue_source_selection_v1",
        "status": "selected",
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: str(row["slug"])),
        "selection_rule": "minimum normalized gate deficit, then maximum bridge score",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
