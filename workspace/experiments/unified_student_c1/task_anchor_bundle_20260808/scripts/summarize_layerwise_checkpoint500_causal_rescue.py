#!/usr/bin/env python3
"""Validate and compare conditional checkpoint-500 causal rescue arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPECS = {
    "dense_same_total": {"repeat": 0.10, "weight": 1.0, "causal_pairs": 100},
    "strong_same_density": {"repeat": 0.05, "weight": 2.0, "causal_pairs": 50},
}


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return value


def one(root: Path, branch: str, name: str) -> Path:
    """Resolve an artifact by its signed branch receipt, not AMLT path spelling."""
    receipts = [
        path
        for path in root.rglob("rescue_training_receipt.json")
        if read(path).get("branch") == branch
    ]
    if len(receipts) != 1:
        raise ValueError(f"Expected one receipt for {branch}, got {receipts}")
    receipt = receipts[0]
    # AMLT stores the receipt under metrics/ and the causal gate under eval/c4/.
    # Unit fixtures may place both directly in the branch directory.  Ascend
    # only as far as the supplied result root and take the nearest unique match,
    # which prevents crossing into the sibling branch.
    ancestors = [receipt.parent]
    ancestors.extend(parent for parent in receipt.parents if parent != receipt.parent)
    for ancestor in ancestors:
        try:
            ancestor.relative_to(root)
        except ValueError:
            break
        matches = sorted(ancestor.rglob(name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            break
        if ancestor == root:
            break
    raise ValueError(f"Expected one {branch}/{name} next to {receipt}, got none")


def summarize(root: Path) -> dict:
    rows = []
    for branch, spec in SPECS.items():
        receipt = read(one(root, branch, "rescue_training_receipt.json"))
        completion = read(one(root, branch, "completion.json"))
        config = read(one(root, branch, "run_config.json"))
        gate = read(one(root, branch, "recurrent_causal_gate.json"))
        summary = gate.get("summary") or {}
        clean = summary.get("clean") or {}
        checks = {
            "registered_branch": receipt.get("branch") == branch,
            "diagnostic_only": receipt.get("diagnostic_only") is True and receipt.get("exact_resume") is False,
            "optimizer_reset": receipt.get("optimizer_reset") is True and config.get("diagnostic_optimizer_reset") is True,
            "source_boundary": (
                receipt.get("source_checkpoint_round") == 500
                and receipt.get("source_world_size") == 16
                and receipt.get("current_world_size") == 4
                and receipt.get("start_global_position") == 8000
                and receipt.get("end_global_position") == 8999
            ),
            "segment_support": (
                completion.get("status") == "diagnostic_complete"
                and completion.get("micro_steps") == 1000
                and completion.get("optimizer_steps") == 250
                and completion.get("history_rows") == 1000
                and completion.get("history_global_position_start") == 8000
                and completion.get("history_global_position_end") == 8999
            ),
            "causal_schedule": (
                completion.get("causal_pairs_seen") == spec["causal_pairs"]
                and float(config.get("causal_repeat_fraction", -1)) == spec["repeat"]
                and float(config.get("causal_weight", -1)) == spec["weight"]
                and config.get("causal_objective_mode") == "specificity_triplet"
            ),
            "closed_layerwise": (
                gate.get("state_architecture") == "layerwise"
                and gate.get("state_shape") == [64, 5120]
                and gate.get("raw_history_visible_to_decoder") is False
                and gate.get("persistent_history_kv_cache") is False
            ),
            "c4_support": summary.get("all_pairs_n") == 150 and int(summary.get("clean_n", 0)) >= 50,
        }
        if not all(checks.values()):
            raise ValueError(f"Invalid rescue arm {branch}: {[key for key,value in checks.items() if not value]}")
        row = {
            "branch": branch,
            "checks": checks,
            "micro_steps": completion["micro_steps"],
            "optimizer_steps": completion["optimizer_steps"],
            "causal_pairs_seen": completion["causal_pairs_seen"],
            "clean_n": summary["clean_n"],
            "swap": clean["swap"]["flip_to_A_rate"],
            "random": clean["random"]["flip_to_A_rate"],
            "same_rule": clean["samemal"]["flip_to_A_rate"],
            "noop": clean["noop"]["flip_to_A_rate"],
            "D_specific": summary["D_specific"],
            "scientific_gate_pass": summary["scientific_gate_pass"],
        }
        rows.append(row)
    eligible = [row for row in rows if row["scientific_gate_pass"]]
    selected = max(eligible, key=lambda row: row["D_specific"]) if eligible else None
    return {
        "stage": "layerwise_checkpoint500_causal_rescue_comparison",
        "diagnostic_only": True,
        "main_exact_epoch_unchanged": True,
        "arms": rows,
        "any_scientific_gate_pass": bool(eligible),
        "selected_branch": selected["branch"] if selected else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
