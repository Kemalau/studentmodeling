from __future__ import annotations

import json
from pathlib import Path

from scripts.select_state8_taxonomy_consolidation import score


T3 = (
    "diagnose_misconception/eedi_mining/open_taxonomy_generation",
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation",
    "diagnose_misconception/malrule/synthetic_same_template_open_generation",
)


def fixture(root: Path, *, c4_pass: bool = True) -> None:
    artifact = root / "download/exp/job"
    (artifact / "eval/formal_t3_300").mkdir(parents=True)
    (artifact / "metrics").mkdir(parents=True)
    (root / "t3_semantic").mkdir(parents=True)
    (root / "c4_semantic").mkdir(parents=True)
    artifacts = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    groups = {
        "predict_correctness/a": {"accuracy": 0.6},
        "predict_response/a": {"math_equivalence": 0.5},
        "predict_prospective_error/a": {"joint_math_equivalence": 0.5},
        "predict_error_persistence/a": {"accuracy": 0.6},
    }
    (artifact / "eval/formal_t3_300/metrics.json").write_text(json.dumps({
        "state_shape": [64, 8, 5120], "raw_history_visible_to_decoder": False,
        "metrics_by_group": groups,
    }))
    (artifact / "metrics/publish_receipt.json").write_text(json.dumps({
        "stage": "state8_stateonly_repair_pilot_publish_v1", "status": "complete",
        "state_shape": [64, 8, 5120], "raw_history_visible_to_decoder": False,
        "train_entrypoint": "scripts/train_state8_taxonomy_causal_repair.py",
        "artifact_sha256": artifacts,
    }))
    (artifact / "metrics/taxonomy_causal_repair.json").write_text(json.dumps({
        "contract": "source_aware_no_answer_labels_v1",
        "candidate_taxonomy_visible_to_decoder": False,
    }))
    (root / "t3_semantic/metrics.json").write_text(json.dumps({
        "judge_backend": "trapi", "judge_errors": 0,
        "metrics_by_group": {name: {
            "llm_judge_n": 25, "llm_judge_accuracy": 0.6,
            "natural_schema_valid_rate": 1.0, "llm_judge_exact_shortcuts": 0,
        } for name in T3},
    }))
    (root / "c4_semantic/summary.json").write_text(json.dumps({
        "semantic_judge_authoritative": True, "pass": c4_pass,
        "semantic_D_specific": 0.4, "same_rule_recipient_retention": 0.8,
        "noop_recipient_retention": 0.9, "random_matches_either_stable_target_rate": 0.1,
        "generation": {"artifact_sha256": artifacts},
    }))


def test_consolidation_selector_requires_unified_tasks_and_semantic_c4(tmp_path: Path) -> None:
    root = tmp_path / "candidate"; fixture(root)
    result = score("taxonomy_balanced", root)
    assert result["qualified"] is True
    assert result["finalfit_variant"] == "taxonomy_curriculum_balanced"


def test_failed_c4_cannot_be_promoted(tmp_path: Path) -> None:
    root = tmp_path / "candidate"; fixture(root, c4_pass=False)
    assert score("taxonomy_t3heavy", root)["qualified"] is False
