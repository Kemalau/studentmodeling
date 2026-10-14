#!/usr/bin/env python3
"""Apply the exact unified promotion gate to conditional rescue candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from select_state8_exact_diagnosis_curriculum import score_candidate
except ImportError:
    from scripts.select_state8_exact_diagnosis_curriculum import score_candidate


CANDIDATES = {
    "semantic_preserve": {
        "job": "state8-exact-rescue-semantic-preserve-seed66",
        "slug": "exact_rescue_semantic_preserve_seed66",
        "stable_root": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed66_trajectory01_"
            "exact_scientific_rescue_semantic_preserve_seed66_v1"
        ),
        "finalfit_variant": "strong1_exact_rescue_semantic_preserve_seed66",
        "finalfit_job": (
            "state8-strong1-exact-rescue-semantic-preserve-seed66-"
            "unified-finalfit-seed42"
        ),
        "entrypoints": ("scripts/train_state8_diagnosis_first_causal_repair.py",),
    },
    "causal_taskanchor": {
        "job": "state8-exact-rescue-causal-taskanchor-seed67",
        "slug": "exact_rescue_causal_taskanchor_seed67",
        "stable_root": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed67_taskanchor_"
            "exact_scientific_rescue_causal_taskanchor_seed67_v1"
        ),
        "finalfit_variant": "strong1_exact_rescue_causal_taskanchor_seed67",
        "finalfit_job": (
            "state8-strong1-exact-rescue-causal-taskanchor-seed67-"
            "unified-finalfit-seed42"
        ),
        "entrypoints": (
            "scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py",
        ),
    },
    "retention_preserve": {
        "job": "state8-exact-rescue-retention-preserve-seed68",
        "slug": "exact_rescue_retention_preserve_seed68",
        "stable_root": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed68_taskanchor_"
            "exact_scientific_rescue_retention_preserve_seed68_v1"
        ),
        "finalfit_variant": "strong1_exact_rescue_retention_preserve_seed68",
        "finalfit_job": (
            "state8-strong1-exact-rescue-retention-preserve-seed68-"
            "unified-finalfit-seed42"
        ),
        "entrypoints": (
            "scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py",
        ),
    },
}


def select(root: Path, modes: list[str]) -> dict:
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("Rescue modes must be unique and non-empty")
    candidates = []
    for mode in modes:
        config = CANDIDATES.get(mode)
        if config is None:
            raise ValueError(f"Unknown rescue mode: {mode}")
        candidate_root = root / mode
        candidates.append(
            score_candidate(
                candidate_root,
                job=config["job"],
                slug=config["slug"],
                stable_root=config["stable_root"],
                finalfit_variant=config["finalfit_variant"],
                finalfit_job=config["finalfit_job"],
                allowed_repair_entrypoints=config["entrypoints"],
            )
        )
    qualified = [row for row in candidates if row["qualified"]]
    selected = max(
        qualified, key=lambda row: (float(row["score"]), str(row["slug"]))
    ) if qualified else None
    return {
        "stage": "state8_exact_scientific_rescue_joint_promotion_v1",
        "pass": selected is not None,
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: str(row["slug"])),
        "selection_scope": (
            "same_exact_strict_state_only_unified_answer_and_diagnosis_c4_gate"
        ),
        "exact_match_authoritative": False,
        "semantic_judge_authoritative": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=tuple(CANDIDATES), action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.root, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
