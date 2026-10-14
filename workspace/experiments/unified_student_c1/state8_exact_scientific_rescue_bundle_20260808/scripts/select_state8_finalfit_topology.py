#!/usr/bin/env python3
"""Pin one State8 final-fit topology before the first remote submission.

The selection is immutable once written.  This prevents an automatic recovery
from mixing world4 and world2 checkpoints, whose global batches match but whose
rank-local dropout RNG streams do not.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TOPOLOGIES = {
    "world4": {
        "gpu_count": 4,
        "gradient_accumulation_steps": 4,
        "distributed_rounds": 64962,
        "config": "amlt/feeds_state8_strong1_widen8_final_fit_20260808.yaml",
        "validator": "scripts/validate_state8_unified_final_fit_amlt.py",
    },
    "world2": {
        "gpu_count": 2,
        "gradient_accumulation_steps": 8,
        "distributed_rounds": 129923,
        "config": "amlt/feeds_state8_strong1_widen8_final_fit_world2_20260808.yaml",
        "validator": "scripts/validate_state8_strong1_finalfit_world2_amlt.py",
    },
}


def select(output: Path, requested: str, available_user_slots: int) -> dict:
    if requested == "auto":
        topology = "world4" if available_user_slots >= 4 else "world2"
    else:
        topology = requested
    expected = TOPOLOGIES[topology]
    result = {
        "stage": "state8_finalfit_topology_selection_v1",
        "status": "pinned",
        "topology": topology,
        "available_user_slots_at_selection": available_user_slots,
        "global_batch_size": 16,
        "optimizer_steps": 16241,
        "real_examples_exactly_once": 259845,
        "rank_local_dropout_realization_changes_with_topology": True,
        "cross_topology_resume_allowed": False,
        **expected,
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != result:
            raise ValueError(
                f"final-fit topology is already pinned: existing={existing} requested={result}"
            )
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology", choices=("auto", "world4", "world2"), default="auto")
    parser.add_argument("--available-user-slots", type=int, required=True)
    args = parser.parse_args()
    if args.available_user_slots < 0:
        raise ValueError("available user slots must be non-negative")
    print(
        json.dumps(
            select(args.output, args.topology, args.available_user_slots),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
