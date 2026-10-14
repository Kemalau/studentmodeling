from __future__ import annotations

import json
from pathlib import Path

from scripts.select_state8_taxonomy_curriculum import score


T3 = (
    "diagnose_misconception/eedi_mining/open_taxonomy_generation",
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation",
    "diagnose_misconception/malrule/synthetic_same_template_open_generation",
)


def test_selector_uses_semantics_and_all_unified_tasks(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    artifact = root / "download/exp/job"
    (root / "t3_semantic").mkdir(parents=True)
    (root / "c4_semantic").mkdir(parents=True)
    (artifact / "eval/formal_screen").mkdir(parents=True)
    (artifact / "metrics").mkdir(parents=True)
    semantic_groups = {
        name: {"llm_judge_n": 25, "llm_judge_accuracy": 0.6,
               "natural_schema_valid_rate": 1.0, "llm_judge_exact_shortcuts": 0}
        for name in T3
    }
    (root / "t3_semantic/metrics.json").write_text(json.dumps({
        "judge_backend": "trapi", "judge_errors": 0, "metrics_by_group": semantic_groups}))
    groups = {
        "predict_correctness/a": {"accuracy": 0.6},
        "predict_response/a": {"math_equivalence": 0.5},
        "predict_prospective_error/a": {"joint_math_equivalence": 0.5},
        "predict_error_persistence/a": {"accuracy": 0.6},
    }
    (artifact / "eval/formal_screen/metrics.json").write_text(json.dumps({
        "state_shape": [64, 8, 5120], "raw_history_visible_to_decoder": False,
        "metrics_by_group": groups}))
    (artifact / "metrics/immutable_receipt.json").write_text(json.dumps({
        "status": "complete", "stage": "test", "state_shape": [64, 8, 5120],
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}}))
    (artifact / "metrics/taxonomy_teacher_curriculum.json").write_text(json.dumps({
        "semantic_judge_authoritative": True, "exact_label_metric_authoritative": False}))
    artifacts = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    (root / "c4_semantic/summary.json").write_text(json.dumps({
        "semantic_judge_authoritative": True,
        "pass": True,
        "semantic_D_specific": 0.5,
        "same_rule_recipient_retention": 0.8,
        "noop_recipient_retention": 0.9,
        "random_matches_either_stable_target_rate": 0.1,
        "generation": {"artifact_sha256": artifacts},
    }))
    result = score("taxonomy_balanced", root)
    assert result["qualified"] is True
    assert set(result["checks"]) == {
        "T1", "T2", "T3_semantic", "T3_schema", "T5", "T6", "C4_pass",
        "C4_D_specific", "C4_same_rule_retention", "C4_noop_retention",
        "C4_random_stable",
    }
    assert result["artifact_sha256"]["adapter_model"] == "a" * 64
