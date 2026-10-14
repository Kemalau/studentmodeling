#!/usr/bin/env python3
"""Combine gold-referenced and native-output-relative dual-C4 semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def summarize(
    receipt: dict,
    malrule_gold: dict,
    eedi_gold: dict,
    malrule_pairwise: dict,
    eedi_pairwise: dict,
) -> dict:
    if receipt.get("stage") != "state8_repair_dual_c4_quickscreen_generation_v1":
        raise ValueError("Generation receipt stage drift")
    if receipt.get("status") != "complete" or receipt.get("diagnostic_only") is not True:
        raise ValueError("Generation receipt is incomplete")
    if receipt.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Raw history leaked to decoder")
    for name, value in (("malrule", malrule_gold), ("eedi", eedi_gold)):
        if value.get("semantic_judge_authoritative") is not True:
            raise ValueError(f"{name} semantic judge authority drift")
        if value.get("exact_label_role") != "auxiliary_only_not_used_for_gate":
            raise ValueError(f"{name} exact-label role drift")
        generation = value.get("generation") or {}
        if generation.get("artifact_sha256") != receipt.get("artifact_sha256"):
            raise ValueError(f"{name} artifact binding drift")
    for name, value in (("malrule", malrule_pairwise), ("eedi", eedi_pairwise)):
        if value.get("semantic_judge_authoritative") is not True:
            raise ValueError(f"{name} pairwise judge authority drift")
        if value.get("stage") != "state8_diagnosis_c4_pairwise_summary_v1":
            raise ValueError(f"{name} pairwise stage drift")
    def compact(gold: dict, pairwise: dict) -> dict:
        cells = gold["cells"]
        return {
            "pass": gold.get("pass") is True,
            "semantic_D_specific": float(gold["semantic_D_specific"]),
            "native_to_recipient": float(cells["native_to_recipient"]["semantic_equivalence_rate"]),
            "swap_to_donor": float(cells["swap_to_donor"]["semantic_equivalence_rate"]),
            "random_matches_either_stable_target": float(gold["random_matches_either_stable_target_rate"]),
            "noop_to_gold_recipient": float(gold["noop_recipient_retention"]),
            "same_rule_to_gold_recipient": float(gold["same_rule_recipient_retention"]),
            "noop_native_output_invariance": float(pairwise["noop_native_output_invariance"]),
            "same_rule_native_output_invariance": float(pairwise["same_rule_native_output_invariance"]),
            "random_native_output_invariance": float(pairwise["random_native_output_invariance"]),
            "swap_native_output_invariance": float(pairwise["swap_native_output_invariance"]),
            "same_rule_vs_swap_separation": float(pairwise["same_rule_vs_swap_separation"]),
        }
    malrule = compact(malrule_gold, malrule_pairwise)
    eedi = compact(eedi_gold, eedi_pairwise)
    score = (
        0.20 * max(0.0, malrule["semantic_D_specific"])
        + 0.25 * max(0.0, eedi["semantic_D_specific"])
        + 0.15 * malrule["swap_to_donor"]
        + 0.20 * eedi["swap_to_donor"]
        + 0.05 * malrule["same_rule_vs_swap_separation"]
        + 0.05 * eedi["same_rule_vs_swap_separation"]
        + 0.05 * malrule["noop_native_output_invariance"]
        + 0.05 * eedi["noop_native_output_invariance"]
        - 0.05 * malrule["random_matches_either_stable_target"]
        - 0.10 * eedi["random_matches_either_stable_target"]
    )
    return {
        "stage": "state8_repair_dual_c4_semantic_quickscreen_v1",
        "status": "complete",
        "diagnostic_only": True,
        "repair_seed": int(receipt["repair_seed"]),
        "repair_branch": str(receipt["repair_branch"]),
        "checkpoint_step": int(receipt["checkpoint_step"]),
        "artifact_sha256": receipt["artifact_sha256"],
        "raw_history_visible_to_decoder": False,
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "malrule": malrule,
        "eedi_same_question": eedi,
        "dual_c4_pass": malrule["pass"] and eedi["pass"],
        "diagnostic_score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--malrule-gold", type=Path, required=True)
    parser.add_argument("--eedi-gold", type=Path, required=True)
    parser.add_argument("--malrule-pairwise", type=Path, required=True)
    parser.add_argument("--eedi-pairwise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        read(args.receipt), read(args.malrule_gold), read(args.eedi_gold),
        read(args.malrule_pairwise), read(args.eedi_pairwise),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
