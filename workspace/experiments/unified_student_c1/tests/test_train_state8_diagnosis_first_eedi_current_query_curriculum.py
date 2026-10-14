from __future__ import annotations

import json

from state8_current_query_routing import apply_current_query_routing


def test_eedi_current_item_is_visible_but_wrong_response_stays_in_state() -> None:
    observation = {
        "question_text": "Compare 3/4 and 5/8\n### Correct: >\n### Incorrect: <",
        "question_id": "q1_C",
    }
    update = (
        "<student_state_update><STUDENT_STATE><observed_interaction>"
        + json.dumps(observation)
        + "</observed_interaction><STUDENT_STATE></student_state_update>"
    )
    row = {
        "example_id": "eedi",
        "task": "diagnose_misconception",
        "source": "eedi_mining",
        "decode_text": "instruction\n<STUDENT_STATE>\n<query>diagnose</query>\nAnswer:",
        "direct_text": (
            "instruction\n<student_history>private wrong answer</student_history>\n"
            "<STUDENT_STATE>\n<query>diagnose</query>\nAnswer:"
        ),
        "update_texts": [update],
    }
    audit = apply_current_query_routing([row], sources={"eedi_mining"})
    query = row["decode_text"].split("<current_item>", 1)[1]
    assert "Compare 3/4 and 5/8" in query
    assert "### Correct" in query
    assert "### Incorrect" not in query
    assert "private wrong answer" not in row["decode_text"]
    assert "private wrong answer" in row["direct_text"]
    assert audit["current_query_routed_rows"] == 1


def test_current_query_curriculum_declares_distinct_scientific_receipt() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "state8_diagnosis_first_eedi_current_query_curriculum_v1",
        "current_item_visible_response_state_only_v1:eedi_only",
        "student_response_visible_to_decoder\": False",
        "raw_history_visible_to_decoder\": False",
        "semantic_judge_authoritative\": True",
        "exact_match_authoritative\": False",
    ):
        assert fragment in source
