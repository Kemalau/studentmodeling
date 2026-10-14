#!/usr/bin/env python3
"""Bind one source-anchor arm to authoritative semantic evaluation outputs."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--formal-metrics", type=Path, required=True)
    parser.add_argument("--t3-semantic", type=Path, required=True)
    parser.add_argument("--diagnosis-c4", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal = read(args.formal_metrics)
    t3 = read(args.t3_semantic)
    diagnosis = read(args.diagnosis_c4)
    receipt = read(args.source_receipt)
    completion = read(args.completion)
    if receipt.get("stage") != "state8_diagnosis_first_source_anchored_repair_v1":
        raise ValueError("Source-anchor receipt stage drift")
    if receipt.get("status") != "complete" or receipt.get("state_token_count") != 8:
        raise ValueError("Source-anchor receipt is incomplete")
    if receipt.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Decoder history leakage")
    if receipt.get("final_inference_route") != "strict_state_only":
        raise ValueError("Final inference route drift")
    if receipt.get("canonical_label_role") != "auxiliary_ordinary_weight":
        raise ValueError("Canonical-label supervision drift")
    if completion.get("status") != "complete":
        raise ValueError("Training completion receipt is incomplete")
    if t3.get("judge_backend") != "trapi" or int(t3.get("judge_errors", -1)) != 0:
        raise ValueError("T3 semantic judgment is not authoritative and complete")
    if diagnosis.get("semantic_judge_authoritative") is not True:
        raise ValueError("Diagnosis-C4 semantic judgment is not authoritative")
    if diagnosis.get("exact_label_role") != "auxiliary_only_not_used_for_gate":
        raise ValueError("Diagnosis-C4 exact-label role drift")
    groups = t3.get("metrics_by_group") or {}
    expected = {
        "diagnose_misconception/eedi_mining/open_taxonomy_generation",
        "diagnose_misconception/malrule/synthetic_cross_template_open_generation",
        "diagnose_misconception/malrule/synthetic_same_template_open_generation",
    }
    if set(groups) != expected:
        raise ValueError(f"T3 semantic group drift: {sorted(groups)}")
    if any(int(row.get("llm_judge_n", -1)) != 25 for row in groups.values()):
        raise ValueError("T3 semantic support drift")
    result = {
        "stage": "state8_source_anchor_candidate_semantic_summary_v1",
        "status": "complete",
        "arm": args.arm,
        "job": args.job,
        "state_token_count": 8,
        "raw_history_visible_to_decoder": False,
        "final_inference_route": "strict_state_only",
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "canonical_label_role": "auxiliary_ordinary_weight",
        "formal_metrics": formal.get("metrics_by_group") or {},
        "t3_metrics_by_group": groups,
        "diagnosis_c4": diagnosis,
        "artifact_sha256": completion.get("sha256") or {},
        "source_anchor_receipt": receipt,
        "answer_c4_confirmation_pending": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
