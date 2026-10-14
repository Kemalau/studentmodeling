from __future__ import annotations

import pytest

from scripts.train_state8_diagnosis_first_causal_repair import (
    LABEL_FIRST_ORDER,
    apply_diagnosis_first_order,
    diagnosis_first_prompt,
    reorder_target,
)
from scripts.t3_judge_utils import natural_schema_valid


LABEL_FIRST = (
    "<label>702</label>\n"
    "<diagnosis>Omits significant zeroes in standard form.</diagnosis>\n"
    "<evidence>The student wrote 3.2 instead of 3.002.</evidence>"
)


@pytest.fixture(autouse=True)
def pin_diagnosis_first_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy diagnosis-first assertions independent of run-level mode."""
    monkeypatch.setenv("STATE8_DIAGNOSIS_FIELD_ORDER", "diagnosis_first")


def test_reorders_semantic_fields_before_auxiliary_label() -> None:
    assert reorder_target(LABEL_FIRST) == (
        "<diagnosis>Omits significant zeroes in standard form.</diagnosis>\n"
        "<evidence>The student wrote 3.2 instead of 3.002.</evidence>\n"
        "<label>702</label>"
    )
    assert natural_schema_valid(reorder_target(LABEL_FIRST))


def test_label_first_order_makes_identity_the_first_free_generation() -> None:
    diagnosis_first = reorder_target(LABEL_FIRST)
    assert reorder_target(diagnosis_first, LABEL_FIRST_ORDER) == LABEL_FIRST
    prompt = diagnosis_first_prompt(
        "Return exactly one non-empty <diagnosis>, <evidence>, and <label> tag, "
        "in that order. The natural-language diagnosis is primary and the "
        "canonical label is auxiliary.",
        LABEL_FIRST_ORDER,
    )
    assert "<label>, <diagnosis>, and <evidence>" in prompt
    assert "Select the misconception label from the latent state" in prompt


def test_reorder_rejects_partial_natural_schema() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        reorder_target("<label>702</label><diagnosis>missing evidence</diagnosis>")


def test_prompt_makes_semantics_primary_without_gold_label() -> None:
    prompt = diagnosis_first_prompt(
        "Return exactly <label>, <diagnosis>, and <evidence> tags."
    )
    assert "<diagnosis>, <evidence>, and <label>" in prompt
    assert "canonical label is auxiliary" in prompt
    assert "702" not in prompt


def test_applies_to_real_and_causal_diagnosis_rows_only() -> None:
    rows = [
        {
            "task": "diagnose_misconception",
            "target": LABEL_FIRST,
            "decode_text": "Return exactly <label>, <diagnosis>, and <evidence> tags.",
        },
        {
            "causal_target_kind": "diagnosis_natural_schema",
            "native_target": LABEL_FIRST,
            "donor_target": LABEL_FIRST.replace("702", "703"),
            "candidate_targets": {"702": LABEL_FIRST},
            "decode_text": "Return exactly one non-empty <label>, <diagnosis>, and <evidence> tag.",
        },
        {"task": "predict_correctness", "target": "<correctness>correct</correctness>"},
    ]
    audit = apply_diagnosis_first_order(rows)
    assert audit == {"diagnosis_first_target_rows": 2, "diagnosis_first_prompt_rows": 2}
    assert rows[0]["target"].startswith("<diagnosis>")
    assert rows[1]["native_target"].endswith("<label>702</label>")
    assert rows[2]["target"] == "<correctness>correct</correctness>"


def test_evaluators_apply_the_same_order_contract() -> None:
    from scripts.evaluate_recurrent_diagnosis_c4_diagnosis_first import prompt, target
    from scripts.evaluate_recurrent_formal_protocols_diagnosis_first import combined

    rows = [{
        "task": "diagnose_misconception",
        "source": "eedi_mining",
        "target": "<misconception>Omits significant zeroes.</misconception>",
        "target_label": "702",
        "decode_text": "Return exactly one valid XML-style tag.\n<STUDENT_STATE>\nAnswer:",
        "direct_text": "Return exactly one valid XML-style tag.\n<student_history>x</student_history>\n<STUDENT_STATE>\nAnswer:",
        "update_texts": [],
    }]
    combined(rows)
    assert rows[0]["target"].startswith("<diagnosis>")
    assert rows[0]["target"].endswith("<label>702</label>")
    assert "<diagnosis>, <evidence>, and <label>" in rows[0]["decode_text"]

    c4_prompt = prompt({
        "example_id": "x",
        "target_label": "12463",
        "target": "<response>12463</response>",
        "query": {"problem_text": "1/2 + 1/3", "correct_answer": "5/6"},
    })
    assert "<diagnosis>, <evidence>, and <label>" in c4_prompt
    assert "For this Malrule diagnosis" in c4_prompt
    assert "For this Eedi diagnosis" not in c4_prompt
    assert "12463" not in c4_prompt
    c4_target = target("fractions.denominator_comparison_error")
    assert c4_target.startswith("<diagnosis>")
    assert c4_target.endswith(
        "<label>fractions.denominator_comparison_error</label>"
    )
