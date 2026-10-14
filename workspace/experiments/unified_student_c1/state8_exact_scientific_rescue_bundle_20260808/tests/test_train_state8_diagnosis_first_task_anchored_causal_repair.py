from __future__ import annotations

from scripts.train_state8_diagnosis_first_task_anchored_causal_repair import (
    effective_anchor_weight,
)
from scripts.train_state8_diagnosis_first_causal_repair import reorder_target


def test_composed_anchor_is_strong_only_for_diagnosis() -> None:
    schedule = (1.0, 0.5, 0.25)
    assert effective_anchor_weight(
        {"task": "diagnose_misconception"}, 0.2, schedule, 0.1
    ) == (0, 1.0)
    assert effective_anchor_weight(
        {"task": "diagnose_misconception"}, 0.6, schedule, 0.1
    ) == (1, 0.5)
    assert effective_anchor_weight(
        {"task": "predict_error_persistence"}, 0.9, schedule, 0.1
    ) == (2, 0.1)


def test_composed_target_keeps_semantics_before_auxiliary_label() -> None:
    target = (
        "<label>fractions.denominator_comparison_error</label>\n"
        "<diagnosis>The student compares denominators as if the larger one "
        "made the fraction larger.</diagnosis>\n"
        "<evidence>They selected 1/8 over 1/3 solely because 8 is larger.</evidence>"
    )
    rendered = reorder_target(target)
    assert rendered.startswith("<diagnosis>")
    assert rendered.endswith(
        "<label>fractions.denominator_comparison_error</label>"
    )
