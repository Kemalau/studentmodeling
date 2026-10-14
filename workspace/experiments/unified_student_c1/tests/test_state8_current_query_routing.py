import copy
import json

import pytest
from pathlib import Path

from scripts.state8_current_query_routing import (
    apply_current_query_routing,
    inject_current_query,
    observed_payload,
)


def update(index: int) -> str:
    payload = {
        "interaction_id": f"i{index}",
        "question_text": (
            f"question {index}\n### Correct: {index + 1}\n### Incorrect: {index}"
        ),
        "student_answer": str(index),
        "correct_answer": str(index + 1),
    }
    return (
        "<student_state_update>\n<STUDENT_STATE>\n"
        f"<observed_interaction>{json.dumps(payload)}</observed_interaction>\n"
        "<STUDENT_STATE>\n</student_state_update>"
    )


def row(depth: int = 2):
    return {
        "example_id": "example",
        "task": "diagnose_misconception",
        "source": "eedi_mining",
        "decode_text": "instruction\n<STUDENT_STATE>\n<query>diagnose</query>\nAnswer:",
        "direct_text": (
            "instruction\n<student_history>private response</student_history>\n"
            "<STUDENT_STATE>\n<query>diagnose</query>\nAnswer:"
        ),
        "update_texts": [update(index) for index in range(depth)],
        "target": "<diagnosis>gold</diagnosis><evidence>x</evidence><label>1</label>",
    }


def test_routes_item_but_keeps_student_response_in_state():
    value = row(2)
    audit = apply_current_query_routing([value])
    assert len(value["update_texts"]) == 2
    assert '"interaction_id"' not in value["decode_text"]
    assert '"student_answer"' not in value["decode_text"]
    assert '"correct_answer":"2"' in value["decode_text"]
    assert "### Incorrect" not in value["decode_text"]
    assert "### Correct" in value["decode_text"]
    assert "<student_history>private response</student_history>" in value["direct_text"]
    assert '"correct_answer":"2"' in value["direct_text"]
    assert "### Incorrect" not in value["direct_text"].split("<current_item>", 1)[1]
    assert value["raw_prior_history_visible_to_decoder"] is False
    assert value["current_item_visible_to_decoder"] is True
    assert value["student_response_visible_to_decoder"] is False
    assert audit["current_query_state_updates"] == 2
    assert audit["current_query_teacher_rows"] == 1


def test_one_update_still_populates_state():
    value = row(1)
    audit = apply_current_query_routing([value])
    assert len(value["update_texts"]) == 1
    assert audit["current_query_state_updates"] == 1


def test_rejects_supervision_leakage():
    payload = {"question_text": "q", "target_label": "secret"}
    text = f"<observed_interaction>{json.dumps(payload)}</observed_interaction>"
    with pytest.raises(ValueError, match="target fields"):
        observed_payload(text)


def test_does_not_modify_non_diagnosis_rows():
    value = row(1)
    value["task"] = "predict_correctness"
    before = copy.deepcopy(value)
    assert apply_current_query_routing([value])["current_query_routed_rows"] == 0
    assert value == before


def test_source_filter_keeps_existing_malrule_query_untouched():
    eedi = row(1)
    malrule = row(1)
    malrule["example_id"] = "malrule"
    malrule["source"] = "malrule"
    before = copy.deepcopy(malrule)
    audit = apply_current_query_routing([eedi, malrule], sources={"eedi_mining"})
    assert audit["current_query_routed_rows"] == 1
    assert audit["current_query_source_eedi_mining"] == 1
    assert malrule == before


def test_query_segment_must_be_unique():
    with pytest.raises(ValueError, match="exactly one query"):
        inject_current_query("no query", {"question_text": "q"})


def test_checkpoint_auditor_knows_query_aligned_arm():
    from scripts.audit_state8_source_anchor_checkpoint import ARMS

    assert ARMS["current_query_updater_mse1"] == {
        "seed": 3072,
        "tag": "source_anchor_updater_mse1_current_query_v1",
        "relative_mse": 1.0,
        "transition_only": True,
        "current_query": True,
    }
