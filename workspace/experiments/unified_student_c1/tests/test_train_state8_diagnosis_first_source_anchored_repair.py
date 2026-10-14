from scripts.train_state8_diagnosis_first_source_anchored_repair import (
    aggregate_causal_teacher_binding,
    current_query_mode,
    env_flag,
    effective_anchor_weight,
    is_bindable_causal_pair,
    task_anchor_kind,
)


SCHEDULES = {
    "eedi_schedule": (1.0, 1.0, 1.0),
    "malrule_schedule": (1.0, 0.75, 0.5),
    "diagnosis_other_schedule": (1.0, 0.5, 0.25),
    "persistence_schedule": (0.5, 0.75, 1.0),
    "other_weight": 0.1,
}


def test_source_and_task_anchor_kinds_are_explicit() -> None:
    assert task_anchor_kind(
        {"task": "diagnose_misconception", "source": "eedi_mining"}
    ) == "diagnosis_eedi"
    assert task_anchor_kind(
        {"task": "diagnose_misconception", "source": "malrule"}
    ) == "diagnosis_malrule"
    assert task_anchor_kind({"task": "predict_error_persistence"}) == "persistence"
    assert task_anchor_kind({"task": "predict_response"}) == "other"


def test_eedi_anchor_stays_strong_and_persistence_increases() -> None:
    row = {"task": "diagnose_misconception", "source": "eedi_mining"}
    assert effective_anchor_weight(row, 0.1, **SCHEDULES) == (
        0,
        "diagnosis_eedi",
        1.0,
    )
    assert effective_anchor_weight(row, 0.9, **SCHEDULES) == (
        2,
        "diagnosis_eedi",
        1.0,
    )
    persistence = {"task": "predict_error_persistence", "source": "dbe_kt22"}
    assert effective_anchor_weight(persistence, 0.1, **SCHEDULES) == (
        0,
        "persistence",
        0.5,
    )
    assert effective_anchor_weight(persistence, 0.9, **SCHEDULES) == (
        2,
        "persistence",
        1.0,
    )


def test_source_only_changes_training_weight_not_decoder_input() -> None:
    eedi = {"task": "diagnose_misconception", "source": "eedi_mining"}
    malrule = {"task": "diagnose_misconception", "source": "malrule"}
    assert effective_anchor_weight(eedi, 0.6, **SCHEDULES)[1:] == (
        "diagnosis_eedi",
        1.0,
    )
    assert effective_anchor_weight(malrule, 0.6, **SCHEDULES)[1:] == (
        "diagnosis_malrule",
        0.75,
    )


def test_current_query_flag_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("STATE8_CURRENT_QUERY_ROUTING", "true")
    assert env_flag("STATE8_CURRENT_QUERY_ROUTING") is True
    monkeypatch.setenv("STATE8_CURRENT_QUERY_ROUTING", "maybe")
    import pytest

    with pytest.raises(ValueError, match="boolean flag"):
        env_flag("STATE8_CURRENT_QUERY_ROUTING")


def test_current_query_mode_supports_source_aware_routing(monkeypatch) -> None:
    monkeypatch.setenv("STATE8_CURRENT_QUERY_ROUTING", "eedi_only")
    assert current_query_mode() == "eedi_only"
    monkeypatch.setenv("STATE8_CURRENT_QUERY_ROUTING", "1")
    assert current_query_mode() == "all"
    monkeypatch.setenv("STATE8_CURRENT_QUERY_ROUTING", "unknown")
    import pytest

    with pytest.raises(ValueError, match="off, all, or eedi_only"):
        current_query_mode()


def test_causal_teacher_binding_requires_both_state_and_direct_views() -> None:
    complete = {
        "causal_target_kind": "diagnosis_natural_schema",
        "native_update_texts": ["native update"],
        "donor_update_texts": ["donor update"],
        "native_direct_text": "native teacher",
        "donor_direct_text": "donor teacher",
        "native_target": "native target",
        "donor_target": "donor target",
    }
    assert is_bindable_causal_pair(complete)
    assert not is_bindable_causal_pair({**complete, "donor_direct_text": ""})
    assert not is_bindable_causal_pair({**complete, "causal_target_kind": "answer"})


def test_causal_teacher_binding_receipt_uses_global_history() -> None:
    receipt = aggregate_causal_teacher_binding(
        [
            {
                "rank": 0,
                "causal_teacher_binding_effective_weight": 0.5,
                "causal_teacher_binding_source": "eedi_mining",
            },
            {
                "rank": 2,
                "causal_teacher_binding_effective_weight": 0.5,
                "causal_teacher_binding_source": "malrule",
            },
            {
                "rank": 1,
                "causal_teacher_binding_loss": 0.0,
                "causal_teacher_binding_source": "unknown",
            },
        ],
        0.25,
    )
    assert receipt["applied"] == 2
    assert receipt["by_source"] == {"eedi_mining": 1, "malrule": 1}
    assert receipt["by_rank"] == {"0": 1, "2": 1}


def test_positive_causal_teacher_binding_fails_without_records() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="produced no history"):
        aggregate_causal_teacher_binding([], 0.25)
