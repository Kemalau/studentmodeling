#!/usr/bin/env python3
"""Select the highest-scoring qualified terminal State8 candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


STAGES = {
    "state8_strong1_widen8_joint_promotion_v1",
    "state8_strong1_rescue_joint_promotion_v1",
    "state8_midpoint_diagnosis_first_joint_promotion_v1",
    "state8_exact_diagnosis_curriculum_joint_promotion_v1",
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage") not in STAGES:
        raise ValueError(f"Unexpected selection artifact: {path}")
    return value


def select(paths: list[Path]) -> dict[str, Any]:
    existing = [path for path in paths if path.is_file() and path.stat().st_size]
    if not existing:
        raise ValueError("No terminal selection artifacts")
    sources = []
    candidates = []
    values: list[tuple[Path, dict[str, Any]]] = []
    for path in existing:
        value = read(path)
        values.append((path, value))
        sources.append(
            {
                "path": str(path),
                "stage": value["stage"],
                "pass": bool(value.get("pass")),
            }
        )
        candidates.extend(value.get("candidates") or [])
    qualified = [row for row in candidates if row.get("qualified") is True]
    selected = max(
        qualified,
        key=lambda row: (float(row["score"]), str(row["slug"])),
    ) if qualified else None
    if selected is None:
        return {
            "stage": "state8_complete_candidate_selection_v1",
            "pass": False,
            "selected": None,
            "candidates": sorted(candidates, key=lambda row: str(row.get("slug"))),
            "sources": sources,
            "selection_rule": "highest_joint_score_after_all_branches_terminal",
            "exact_match_authoritative": False,
            "semantic_judge_authoritative": True,
        }
    owner = [
        (path, value)
        for path, value in values
        if any(
            row.get("slug") == selected.get("slug")
            and float(row.get("score", float("-inf"))) == float(selected["score"])
            for row in value.get("candidates") or []
        )
    ]
    if len(owner) != 1:
        raise ValueError(f"Selected candidate ownership drift: {selected}")
    result = copy.deepcopy(owner[0][1])
    result["pass"] = True
    result["selected"] = selected
    result["candidates"] = sorted(candidates, key=lambda row: str(row.get("slug")))
    result["complete_selection_audit"] = {
        "stage": "state8_complete_candidate_selection_v1",
        "selected_source": str(owner[0][0]),
        "sources": sources,
        "selection_rule": "highest_joint_score_after_all_branches_terminal",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
