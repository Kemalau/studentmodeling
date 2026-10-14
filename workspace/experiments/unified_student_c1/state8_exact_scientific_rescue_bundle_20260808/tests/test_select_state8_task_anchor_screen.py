from __future__ import annotations

import json
from pathlib import Path

from scripts.select_state8_task_anchor_screen import score


T3 = (
    "diagnose_misconception/eedi_mining/open_taxonomy_generation",
    "diagnose_misconception/malrule/synthetic_cross_template_open_generation",
    "diagnose_misconception/malrule/synthetic_same_template_open_generation",
)


def test_task_anchor_selector_reads_the_shared_preserved_download(tmp_path: Path) -> None:
    state = tmp_path / "state"
    arm = state / "base"
    artifact = state / "download/exp/state8-task-anchor-base-seed56"
    (artifact / "eval/formal_t3_300").mkdir(parents=True)
    (artifact / "metrics").mkdir(parents=True)
    (arm / "t3_semantic").mkdir(parents=True)
    (arm / "c4_semantic").mkdir(parents=True)
    hashes = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    (artifact / "eval/formal_t3_300/metrics.json").write_text(json.dumps({
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "metrics_by_group": {
            "predict_correctness/a": {"accuracy": 0.6},
            "predict_response/a": {"math_equivalence": 0.5},
            "predict_prospective_error/a": {"joint_math_equivalence": 0.5},
            "predict_error_persistence/a": {
                "accuracy": 0.6, "macro_f1": 0.6,
                "positive_recall": 0.5, "n": 25,
            },
        },
    }))
    (artifact / "metrics/publish_receipt.json").write_text(json.dumps({
        "stage": "state8_stateonly_repair_pilot_publish_v1",
        "train_entrypoint": "scripts/train_state8_task_anchored_causal_repair.py",
        "artifact_sha256": hashes,
    }))
    (artifact / "metrics/task_anchored_causal_repair.json").write_text(json.dumps({
        "stage": "state8_task_anchored_taxonomy_causal_repair_v1",
        "raw_history_visible_to_decoder": False,
    }))
    (artifact / "metrics/taxonomy_causal_repair.json").write_text(json.dumps({
        "contract": "source_aware_no_answer_labels_v1",
    }))
    (arm / "t3_semantic/metrics.json").write_text(json.dumps({
        "judge_backend": "trapi",
        "judge_errors": 0,
        "metrics_by_group": {name: {
            "llm_judge_n": 25,
            "llm_judge_accuracy": 0.6,
            "natural_schema_valid_rate": 1.0,
            "llm_judge_exact_shortcuts": 0,
        } for name in T3},
    }))
    (arm / "c4_semantic/summary.json").write_text(json.dumps({
        "pass": True,
        "semantic_judge_authoritative": True,
        "semantic_D_specific": 0.4,
        "same_rule_recipient_retention": 0.8,
        "noop_recipient_retention": 0.9,
        "random_matches_either_stable_target_rate": 0.1,
        "generation": {"artifact_sha256": hashes},
    }))
    result = score("base", arm)
    assert result["qualified"] is True
    assert result["finalfit_variant"] == "task_anchor_base"
    assert result["finalfit_job"] == "state8-task-anchor-base-unified-finalfit-seed42"


def test_task_anchor_selector_rejects_low_t6_macro_f1(tmp_path: Path) -> None:
    test_task_anchor_selector_reads_the_shared_preserved_download(tmp_path)
    state = tmp_path / "state"
    metrics = next(state.glob("download/**/eval/formal_t3_300/metrics.json"))
    value = json.loads(metrics.read_text())
    value["metrics_by_group"]["predict_error_persistence/a"]["macro_f1"] = 0.1
    metrics.write_text(json.dumps(value))
    result = score("base", state / "base")
    assert result["checks"]["T6_macro_f1"] < 0.30
    assert result["qualified"] is False
