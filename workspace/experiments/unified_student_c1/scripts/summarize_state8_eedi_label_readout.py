#!/usr/bin/env python3
"""Compare early-label State8 readout against a no-early-label control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPECS = {
    "early-label-low-cp20",
    "early-label-high-cp20",
    "causal-teacher-low-cp100-control",
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def compact(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("stage") != "state8_eedi_label_readout_c4_v1":
        raise ValueError("Label-readout stage drift")
    if value.get("status") != "complete" or value.get("diagnostic_only") is not True:
        raise ValueError("Label-readout result is incomplete")
    if value.get("semantic_judge_authoritative") is not True:
        raise ValueError("Semantic judge authority drift")
    if value.get("exact_identifier_authoritative") is not False:
        raise ValueError("Exact identifiers became authoritative")
    if value.get("state_shape") != [64, 8, 5120]:
        raise ValueError("State shape drift")
    if value.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Raw history leaked to decoder")
    if value.get("student_response_visible_to_decoder") is not False:
        raise ValueError("Student response leaked to decoder")
    if value.get("target_candidates_visible_to_decoder") is not False:
        raise ValueError("Target candidates leaked to decoder")
    if int(value.get("num_pairs", -1)) != 25:
        raise ValueError("Frozen pair support drift")
    tag = str(value.get("readout_tag") or "")
    if tag not in SPECS:
        raise ValueError(f"Unknown readout tag: {tag}")
    return {
        "readout_tag": tag,
        "diagnostic_pass": value.get("diagnostic_pass") is True,
        "D_specific": float(value["D_specific"]),
        "swap_to_donor_rate": float(value["swap_to_donor_rate"]),
        "native_recipient_retention": float(value["native_recipient_retention"]),
        "noop_recipient_retention": float(value["noop_recipient_retention"]),
        "same_rule_recipient_retention": float(value["same_rule_recipient_retention"]),
        "random_matches_either_target_rate": float(value["random_matches_either_target_rate"]),
        "valid_identifier_rate": float(
            value["by_condition"]["swap"]["valid_identifier_rate"]
        ),
        "artifact_sha256": value["artifact_sha256"],
    }


def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [compact(value) for value in values]
    by_tag = {row["readout_tag"]: row for row in rows}
    if set(by_tag) != SPECS or len(rows) != len(SPECS):
        raise ValueError(f"Readout matrix drift: {sorted(by_tag)}")
    control = by_tag["causal-teacher-low-cp100-control"]
    early = [by_tag["early-label-low-cp20"], by_tag["early-label-high-cp20"]]
    best = max(
        early,
        key=lambda row: (
            bool(row["diagnostic_pass"]),
            row["D_specific"],
            row["swap_to_donor_rate"],
            row["noop_recipient_retention"],
            row["readout_tag"],
        ),
    )
    improvement = {
        key: float(best[key]) - float(control[key])
        for key in (
            "D_specific",
            "swap_to_donor_rate",
            "native_recipient_retention",
            "noop_recipient_retention",
            "same_rule_recipient_retention",
        )
    }
    if best["diagnostic_pass"]:
        diagnosis = "state_identity_decodable_decoder_interface_is_next_bottleneck"
        next_action = "promote_label_first_state_readout_then_generate_semantic_explanation"
    elif improvement["swap_to_donor_rate"] >= 0.20 and best["D_specific"] >= 0.15:
        diagnosis = "partial_state_identity_decoding_needs_joint_label_and_explanation_consolidation"
        next_action = "train_label_first_joint_readout_with_stronger_native_same_rule_retention"
    else:
        diagnosis = "misconception_identity_not_reliably_decodable_from_state"
        next_action = "repair_updater_binding_before_two_stage_explanation"
    return {
        "stage": "state8_eedi_label_readout_comparison_v1",
        "status": "complete",
        "diagnostic_only": True,
        "semantic_judge_authoritative": True,
        "exact_identifier_authoritative": False,
        "best_early_label_tag": best["readout_tag"],
        "best_early_label": best,
        "control": control,
        "improvement_over_control": improvement,
        "interface_diagnosis": diagnosis,
        "next_action": next_action,
        "candidates": sorted(rows, key=lambda row: row["readout_tag"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([read(path) for path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
