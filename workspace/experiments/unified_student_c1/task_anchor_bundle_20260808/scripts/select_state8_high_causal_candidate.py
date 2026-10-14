#!/usr/bin/env python3
"""Select a high-causal State8 arm without trading away semantic diagnosis."""

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


def metrics(
    root: Path, *, expected_entrypoint: str, require_taxonomy: bool = False
) -> dict[str, Any]:
    c4 = read(root / "c4_semantic/summary.json")
    t3 = read(root / "t3_semantic/metrics.json")
    publish = read(root / "publish_receipt.json")
    if c4.get("semantic_judge_authoritative") is not True:
        raise ValueError(f"C4 lacks authoritative semantic judgment: {root}")
    if t3.get("judge_backend") != "trapi" or int(t3.get("judge_errors", -1)) != 0:
        raise ValueError(f"T3 lacks complete TRAPI semantic judgment: {root}")
    if int(t3.get("generation_rows", -1)) != 300 or int(t3.get("judged_rows", -1)) != 300:
        raise ValueError(f"T3 screen support drift: {root}")
    if any(
        int(value.get("llm_judge_exact_shortcuts", -1)) != 0
        for value in (t3.get("metrics_by_group") or {}).values()
    ):
        raise ValueError(f"T3 exact-match authority is forbidden: {root}")
    generation_artifacts = (c4.get("generation") or {}).get("artifact_sha256")
    publish_artifacts = publish.get("artifact_sha256")
    if not isinstance(generation_artifacts, dict) or generation_artifacts != publish_artifacts:
        raise ValueError(f"C4/publish artifact binding drift: {root}")
    if (
        publish.get("status") != "complete"
        or publish.get("stage") != "state8_stateonly_repair_pilot_publish_v1"
        or publish.get("state_shape") != [64, 8, 5120]
        or publish.get("raw_history_visible_to_decoder") is not False
    ):
        raise ValueError(f"Candidate publish contract drift: {root}")
    if publish.get("train_entrypoint") != expected_entrypoint:
        raise ValueError(f"Candidate training entrypoint drift: {root}")
    taxonomy = None
    if require_taxonomy:
        taxonomy = read(root / "taxonomy_causal_repair.json")
        if (
            taxonomy.get("stage") != "state8_taxonomy_causal_repair_v1"
            or taxonomy.get("status") != "complete"
            or taxonomy.get("contract") != "source_aware_no_answer_labels_v1"
            or taxonomy.get("candidate_taxonomy_visible_to_decoder") is not False
            or taxonomy.get("raw_history_visible_to_decoder") is not False
        ):
            raise ValueError(f"Taxonomy candidate contract drift: {root}")
    groups = t3["metrics_by_group"]
    cross = groups[
        "diagnose_misconception/malrule/synthetic_cross_template_open_generation"
    ]["llm_judge_accuracy"]
    same = groups[
        "diagnose_misconception/malrule/synthetic_same_template_open_generation"
    ]["llm_judge_accuracy"]
    eedi = groups[
        "diagnose_misconception/eedi_mining/open_taxonomy_generation"
    ]["llm_judge_accuracy"]
    return {
        "semantic_D_specific": float(c4["semantic_D_specific"]),
        "same_rule_recipient_retention": float(c4["same_rule_recipient_retention"]),
        "random_matches_either_stable_target_rate": float(
            c4["random_matches_either_stable_target_rate"]
        ),
        "t3_cross": float(cross),
        "t3_same": float(same),
        "t3_eedi": float(eedi),
        "c4_pass": bool(c4["pass"]),
        "publish_receipt": publish,
        "taxonomy_contract": taxonomy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--updater", type=Path, required=True)
    parser.add_argument("--mid", type=Path)
    parser.add_argument("--eedi", type=Path)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    semantic_entrypoint = "scripts/train_state8_semantic_causal_repair.py"
    arms = {
        "joint": metrics(args.joint, expected_entrypoint=semantic_entrypoint),
        "updater": metrics(args.updater, expected_entrypoint=semantic_entrypoint),
    }
    if args.mid is not None:
        arms["mid"] = metrics(args.mid, expected_entrypoint=semantic_entrypoint)
    if args.eedi is not None:
        arms["eedi"] = metrics(args.eedi, expected_entrypoint=semantic_entrypoint)
    if args.taxonomy is not None:
        arms["taxonomy"] = metrics(
            args.taxonomy,
            expected_entrypoint="scripts/train_state8_taxonomy_causal_repair.py",
            require_taxonomy=True,
        )
    for value in arms.values():
        value["eligible"] = (
            value["c4_pass"]
            and value["semantic_D_specific"] >= 0.15
            and value["same_rule_recipient_retention"] >= 0.30
            and value["random_matches_either_stable_target_rate"] <= 0.30
            and value["t3_cross"] >= 0.45
            and value["t3_same"] >= 0.30
            and value["t3_eedi"] >= 0.17
        )
        value["score"] = (
            2.0 * value["semantic_D_specific"]
            + value["same_rule_recipient_retention"]
            - value["random_matches_either_stable_target_rate"]
            + 0.5 * value["t3_cross"]
            + 0.25 * value["t3_same"]
            + value["t3_eedi"]
        )
    eligible = [name for name, value in arms.items() if value["eligible"]]
    selected = max(eligible, key=lambda name: arms[name]["score"]) if eligible else None
    result = {
        "stage": "state8_high_causal_candidate_selection_v1",
        "pass": selected is not None,
        "selected": selected,
        "arms": arms,
        "gates": {
            "semantic_D_specific_min": 0.15,
            "same_rule_recipient_retention_min": 0.30,
            "random_matches_either_stable_target_rate_max": 0.30,
            "t3_cross_min": 0.45,
            "t3_same_min": 0.30,
            "t3_eedi_min": 0.17,
            "authoritative_semantic_c4_pass_required": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
