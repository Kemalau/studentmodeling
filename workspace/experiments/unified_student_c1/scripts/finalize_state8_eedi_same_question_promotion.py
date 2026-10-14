#!/usr/bin/env python3
"""Combine same-question diagnostic selection with frozen answer C4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def finalize(selection: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    if selection.get("stage") not in {
        "state8_eedi_same_question_joint_diagnostic_selection_v1",
        "state8_curriculum_checkpoint_joint_selection_v1",
        "state8_global_joint_diagnostic_selection_v1",
    }:
        raise ValueError("Selection stage drift")
    selected = selection.get("selected") or {}
    if selection.get("pass") is not True or selected.get("qualified") is not True:
        raise ValueError("No diagnostic-qualified candidate")
    if answer.get("state_shape") != [64, 8, 5120]:
        raise ValueError("Answer C4 state shape drift")
    if answer.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Answer C4 history leakage")
    if answer.get("persistent_history_kv_cache") is not False:
        raise ValueError("Answer C4 KV leakage")
    if answer.get("pair_seed") != 42 or (answer.get("summary") or {}).get("all_pairs_n") != 150:
        raise ValueError("Frozen answer C4 panel drift")
    if answer.get("artifact_sha256") != selected.get("artifact_sha256"):
        raise ValueError("Answer C4 artifact mismatch")
    summary = answer["summary"]
    clean = summary["clean"]
    answer_checks = {
        "clean_n": int(summary["clean_n"]),
        "swap": float(clean["swap"]["flip_to_A_rate"]),
        "random": float(clean["random"]["flip_to_A_rate"]),
        "same_rule": float(clean["samemal"]["flip_to_A_rate"]),
        "noop": float(clean["noop"]["flip_to_A_rate"]),
        "D_specific": float(summary["D_specific"]),
        "scientific_gate_pass": bool(summary["scientific_gate_pass"]),
    }
    passed = bool(answer_checks["scientific_gate_pass"])
    promoted = {
        **selected,
        "answer_C4_pass": passed,
        "answer_C4": answer_checks,
        "qualified": bool(selected["qualified"] and passed),
    }
    return {
        "stage": "state8_eedi_same_question_joint_promotion_v1",
        "status": "complete",
        "pass": promoted["qualified"],
        "selected": promoted if promoted["qualified"] else None,
        "candidate": promoted,
        "artifact_sha256": selected["artifact_sha256"],
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "state_token_count": 8,
        "raw_history_visible_to_decoder": False,
        "full_exact_epoch_pending": promoted["qualified"],
        "full_multiseed_C4_pending": promoted["qualified"],
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
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
