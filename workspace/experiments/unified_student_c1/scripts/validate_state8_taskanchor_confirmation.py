#!/usr/bin/env python3
"""Joint full-support gate for a pre-final-fit task-anchor milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from validate_state8_final_eval import FROZEN_GROUP_SUPPORT, REFERENCES, weighted
except ImportError:  # package-style unit tests
    from scripts.validate_state8_final_eval import (
        FROZEN_GROUP_SUPPORT,
        REFERENCES,
        weighted,
    )


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def validate(
    formal_path: Path,
    c4_path: Path,
    semantic_path: Path,
    diagnosis_path: Path,
    step: int,
    adapter_sha256: str,
    state_sha256: str,
) -> dict[str, Any]:
    formal, c4, semantic, diagnosis = map(read, (formal_path, c4_path, semantic_path, diagnosis_path))
    groups = formal.get("metrics_by_group") or {}
    artifacts = {"adapter_model": adapter_sha256, "recurrent_state": state_sha256}
    diagnosis_rows = diagnosis.get("rows") or []
    checks = {
        "milestone_step": step in {200, 400, 600, 800, 1000},
        "artifact_binding": formal.get("artifact_sha256") == artifacts
        and all((row.get("generation") or {}).get("artifact_sha256") == artifacts for row in diagnosis_rows),
        "state_shape": formal.get("state_shape") == [64, 8, 5120],
        "history_closed": formal.get("raw_history_visible_to_decoder") is False
        and formal.get("persistent_history_kv_cache") is False,
        "formal_support": int(formal.get("sampled_rows", -1)) == 21676,
        "all_groups": set(groups) == set(FROZEN_GROUP_SUPPORT),
        "group_support": all(name in groups and int(groups[name].get("n", -1)) == support for name, support in FROZEN_GROUP_SUPPORT.items()),
        "semantic_T3": semantic.get("stage") == "final_T3_semantic_judge"
        and semantic.get("pass") is True
        and int(semantic.get("support", -1)) == 1494,
        "semantic_backend": semantic.get("judge_backend") == "trapi",
        "C4_replication": c4.get("stage") == "state8_promotion_c4_multiseed_v1"
        and c4.get("replication_gate_pass") is True
        and float(c4.get("mean_D_specific", -1)) >= 0.15,
        "C4_controls": sum(float(row.get("random", 1)) <= 0.25 for row in c4.get("rows", [])) >= 2
        and sum(float(row.get("noop", 1)) <= 0.05 for row in c4.get("rows", [])) >= 2
        and sum(float(row.get("same_rule", 1)) <= 0.10 for row in c4.get("rows", [])) >= 2,
        "diagnosis_C4_replication": diagnosis.get("stage") == "state8_closed_diagnosis_c4_semantic_multiseed_v1"
        and diagnosis.get("semantic_judge_authoritative") is True
        and diagnosis.get("exact_label_role") == "auxiliary_only_not_used_for_gate"
        and diagnosis.get("pass") is True
        and int(diagnosis.get("seed_passes", -1)) >= 2,
        "diagnosis_C4_specificity": float(diagnosis.get("mean_semantic_D_specific", -1)) >= 0.15
        and float(diagnosis.get("mean_swap_to_donor", -1)) >= 0.50
        and float(diagnosis.get("mean_random_matches_either", 1)) <= 0.25,
    }
    performance: dict[str, Any] = {}
    for task, (metric, reference, floor) in REFERENCES.items():
        row = weighted(groups, task, metric)
        row.update(reference=reference, minimum_retention_ratio=floor, retention_ratio=row["score"] / reference)
        row["pass"] = row["retention_ratio"] >= floor
        performance[task] = row
    t6_recall = weighted(groups, "predict_error_persistence", "positive_recall")["score"]
    performance["predict_error_persistence"]["positive_recall"] = t6_recall
    performance["predict_error_persistence"]["positive_recall_pass"] = t6_recall > 0
    checks["T6_positive_recall"] = t6_recall > 0
    checks["non_T3_task_retention"] = all(row["pass"] for row in performance.values())
    return {
        "stage": "state8_taskanchor_full_confirmation_v1",
        "status": "complete",
        "pass": all(checks.values()),
        "checkpoint_step": step,
        "checks": checks,
        "performance": performance,
        "T3_semantic": semantic,
        "C4": c4,
        "diagnosis_C4": diagnosis,
        "artifact_sha256": artifacts,
        "state_token_count": 8,
        "final_inference_route": "strict_state_only",
        "semantic_judge_authoritative": True,
        "exact_match_authoritative": False,
        "requires_exact_one_epoch_final_fit": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--c4", type=Path, required=True)
    parser.add_argument("--semantic-gate", type=Path, required=True)
    parser.add_argument("--diagnosis-c4", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.formal, args.c4, args.semantic_gate, args.diagnosis_c4, args.step, args.adapter_sha256, args.state_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
