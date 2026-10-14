from __future__ import annotations

import pytest

from scripts.train_state8_diagnosis_first_causal_repair_exact_resume import (
    gather_without_single_resume_duplication,
)


def test_single_device_resume_deduplicates_only_an_exact_prefix() -> None:
    class Context:
        enabled = False

    def original(local_history, history_base, _context):
        return list(history_base) + list(local_history)

    prefix = [{"global_position": 0, "example_id": "a"}]
    suffix = [{"global_position": 1, "example_id": "b"}]
    assert gather_without_single_resume_duplication(
        original, prefix + suffix, prefix, Context()
    ) == prefix + suffix

    drifted = [{"global_position": 0, "example_id": "wrong"}] + suffix
    with pytest.raises(RuntimeError, match="resume history prefix drift"):
        gather_without_single_resume_duplication(
            original, drifted, prefix, Context()
        )


def test_distributed_resume_keeps_history_base_semantics() -> None:
    class Context:
        enabled = True

    def original(local_history, history_base, _context):
        return list(history_base) + list(local_history)

    prefix = [{"global_position": 0, "example_id": "a"}]
    suffix = [{"global_position": 1, "example_id": "b"}]
    assert gather_without_single_resume_duplication(
        original, suffix, prefix, Context()
    ) == prefix + suffix
