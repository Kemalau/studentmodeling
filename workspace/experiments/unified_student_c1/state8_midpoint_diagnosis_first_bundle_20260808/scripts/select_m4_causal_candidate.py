#!/usr/bin/env python3
"""Select the strongest eligible M4 causal diagnostic across experiment families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage") not in {
        "M4_causal_strength_round500",
        "M4_CF4_multiquery_diagnostic",
    }:
        raise ValueError(f"Invalid M4 causal comparison: {path}")
    return value


def eligible(row: dict) -> bool:
    return (
        int(row.get("C4_clean_n", 0)) >= 50
        and float(row.get("C4_D_specific", -1.0)) >= 0.15
        and float(row.get("C4_noop_flip", 1.0)) <= 0.05
        and float(row.get("C4_same_rule_flip", 1.0)) <= 0.10
        and float(row.get("validation_accuracy_gap", -1.0)) >= -0.05
    )


def select(candidates: list[tuple[str, Path]]) -> dict:
    rows = []
    for family, path in candidates:
        comparison = load(path)
        for raw in comparison.get("branches") or []:
            row = dict(raw)
            row["family"] = family
            row["comparison"] = str(path)
            row["eligible"] = eligible(row)
            rows.append(row)
    ranked = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (
            float(row["C4_D_specific"]),
            float(row.get("validation_recurrent_accuracy", 0.0)),
            -float(row.get("C4_D_damage", 1.0)),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked else None
    return {
        "stage": "M4_causal_candidate_selection",
        "scientific_gate_pass": selected is not None,
        "selected_family": selected["family"] if selected else None,
        "selected_branch": selected["branch"] if selected else None,
        "selected_row": selected,
        # A round-500 gate is only a screening signal.  Preserve the complete
        # deterministic ranking so the controller can advance to the next
        # scientifically eligible branch if causal specificity or task
        # quality collapses during the exact one-epoch continuation.
        "ranked_candidates": ranked,
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate comparison as FAMILY=/path/to/comparison.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for value in args.candidate:
        family, separator, path = value.partition("=")
        if not separator or not family or not path:
            raise ValueError(f"Invalid candidate specification: {value}")
        candidates.append((family, Path(path)))
    payload = select(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
