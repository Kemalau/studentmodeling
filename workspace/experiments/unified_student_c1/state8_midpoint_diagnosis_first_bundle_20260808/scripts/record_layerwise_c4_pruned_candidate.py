#!/usr/bin/env python3
"""Record a candidate pruned by an audited frozen full-C4 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return value


def build(evidence: dict, *, candidate_id: str, loss_profile: str,
          learning_rate: float, checkpoint_round: int) -> dict:
    if evidence.get("stage") != "layerwise_checkpoint_local_c4":
        raise ValueError("Unexpected C4 evidence stage")
    if int(evidence.get("checkpoint_round", -1)) != checkpoint_round:
        raise ValueError("C4 checkpoint drift")
    if int(evidence.get("all_pairs_n", -1)) != 150:
        raise ValueError("C4 support drift")
    if int(evidence.get("clean_n", 0)) <= 0:
        raise ValueError("C4 clean support missing")
    if evidence.get("scientific_gate_pass") is not False:
        raise ValueError("Candidate is not a proven C4 failure")
    checks = evidence.get("checks") or {}
    if checks.get("specificity") is not False:
        raise ValueError("Specificity failure is not proven")
    d_specific = float(evidence["D_specific"])
    if d_specific >= 0.15:
        raise ValueError("C4 specificity threshold is not failed")
    return {
        "stage": "layerwise_single_task_retention_candidate_pruned_v1",
        "candidate_id": candidate_id,
        "loss_profile": loss_profile,
        "learning_rate": learning_rate,
        "checkpoint_round": checkpoint_round,
        "all_gates_pass": False,
        "pruned": True,
        "prune_reason": "frozen_full_c4_specificity_gate_failed",
        "C4": {
            "D_specific": d_specific,
            "D_damage": float(evidence["D_damage"]),
            "clean_n": int(evidence["clean_n"]),
            "all_pairs_n": int(evidence["all_pairs_n"]),
            "swap": float(evidence["swap"]),
            "random": float(evidence["random"]),
            "same_rule": float(evidence["same_rule"]),
            "noop": float(evidence["noop"]),
        },
        "checks": {"C4": False, "specificity": False},
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--loss-profile", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--checkpoint-round", type=int, required=True)
    args = parser.parse_args()
    value = build(
        load(args.evidence),
        candidate_id=args.candidate_id,
        loss_profile=args.loss_profile,
        learning_rate=args.learning_rate,
        checkpoint_round=args.checkpoint_round,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
