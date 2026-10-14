#!/usr/bin/env python3
"""Combine the diagnostic selection with its frozen 150-pair answer-C4 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def finalize(selection: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    if selection.get("stage") != "state8_source_anchor_diagnostic_selection_v1":
        raise ValueError("Selection stage drift")
    arm = selection.get("selected_for_answer_c4_confirmation")
    if not selection.get("diagnostic_gate_pass") or arm not in (selection.get("candidates") or {}):
        raise ValueError("No diagnostic-qualified arm")
    candidate = selection["candidates"][arm]
    if answer.get("state_shape") != [64, 8, 5120]:
        raise ValueError("Answer-C4 state shape drift")
    if answer.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Answer-C4 decoder history leakage")
    if answer.get("persistent_history_kv_cache") is not False:
        raise ValueError("Answer-C4 persistent history leakage")
    if answer.get("pair_seed") != 42 or (answer.get("summary") or {}).get("all_pairs_n") != 150:
        raise ValueError("Answer-C4 frozen panel drift")
    if answer.get("artifact_sha256") != candidate.get("artifact_sha256"):
        raise ValueError("Answer-C4 artifact mismatch")
    summary = answer["summary"]
    clean = summary["clean"]
    checks = {
        "clean_n": int(summary["clean_n"]),
        "swap": float(clean["swap"]["flip_to_A_rate"]),
        "random": float(clean["random"]["flip_to_A_rate"]),
        "same_rule": float(clean["samemal"]["flip_to_A_rate"]),
        "noop": float(clean["noop"]["flip_to_A_rate"]),
        "D_specific": float(summary["D_specific"]),
        "scientific_gate_pass": bool(summary["scientific_gate_pass"]),
    }
    joint = bool(selection["diagnostic_gate_pass"] and checks["scientific_gate_pass"])
    return {
        "stage": "state8_source_anchor_joint_pilot_gate_v1",
        "status": "complete",
        "joint_pilot_gate_pass": joint,
        "selected_arm": arm if joint else None,
        "candidate_arm": arm,
        "candidate_job": candidate["job"],
        "artifact_sha256": candidate["artifact_sha256"],
        "state_token_count": 8,
        "raw_history_visible_to_decoder": False,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "diagnostic_checks": candidate["checks"],
        "diagnostic_floors": candidate["floors"],
        "answer_c4": checks,
        "full_multiseed_confirmation_pending": joint,
        "selection_rule": "diagnostic semantic joint gate plus frozen 150-pair seed-42 answer-C4; full multi-seed confirmation remains required",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--answer-c4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(read(args.selection), read(args.answer_c4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["joint_pilot_gate_pass"] else 2)


if __name__ == "__main__":
    main()
