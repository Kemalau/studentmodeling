from __future__ import annotations

from scripts.train_state8_diagnosis_first_retention_anchored_causal_repair import (
    effective_anchor_weight,
    task_kind,
)


def test_retention_anchor_is_task_and_phase_conditional() -> None:
    diagnosis = (1.0, 0.5, 0.25)
    persistence = (0.5, 0.75, 1.0)
    assert effective_anchor_weight(
        {"task": "diagnose_misconception"}, 0.2, diagnosis, persistence, 0.1
    ) == (0, "diagnosis", 1.0)
    assert effective_anchor_weight(
        {"task": "predict_error_persistence"}, 0.6, diagnosis, persistence, 0.1
    ) == (1, "persistence", 0.75)
    assert effective_anchor_weight(
        {"task": "predict_error_persistence"}, 0.9, diagnosis, persistence, 0.1
    ) == (2, "persistence", 1.0)
    assert effective_anchor_weight(
        {"task": "predict_correctness"}, 0.9, diagnosis, persistence, 0.1
    ) == (2, "other", 0.1)


def test_task_kind_covers_unified_routes() -> None:
    assert task_kind({"task": "diagnose_misconception"}) == "diagnosis"
    assert task_kind({"task": "predict_error_persistence"}) == "persistence"
    assert task_kind({"task": "predict_response"}) == "other"
