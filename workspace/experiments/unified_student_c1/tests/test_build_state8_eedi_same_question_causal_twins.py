import json

from scripts.build_state8_eedi_same_question_causal_twins import (
    build_twins,
    exclusion_sets,
)
from scripts.t3_judge_utils import natural_schema_valid


def update(qid: str, wrong: str, correct: str = "4") -> str:
    payload = {
        "question_id": qid,
        "question_text": f"### Question: 2+2?\n### Correct: {correct}\n### Incorrect: {wrong}",
    }
    return (
        "<student_state_update><STUDENT_STATE><observed_interaction>"
        f"{json.dumps(payload)}</observed_interaction><STUDENT_STATE></student_state_update>"
    )


def row(qid: str, label: str, diagnosis: str, wrong: str):
    return {
        "example_id": f"e:{qid}", "task": "diagnose_misconception", "source": "eedi_mining",
        "target_label": label, "target": f"<misconception>{diagnosis}</misconception>",
        "original_metadata": {"question_id": qid}, "update_texts": [update(qid, wrong)],
        "decode_text": "instruction\n<STUDENT_STATE>\n<query>diagnose</query>\nAnswer:",
        "direct_text": "instruction\n<student_history>private</student_history>\n<STUDENT_STATE>\n<query>diagnose</query>\nAnswer:",
    }


def test_builds_identical_query_swap_and_same_label_control():
    corpus = [
        row("10_A", "1", "adds instead of multiplying", "5"),
        row("10_B", "2", "subtracts instead of adding", "0"),
        row("11_C", "1", "adds instead of multiplying", "6"),
    ]
    output, receipt = build_twins(corpus, twins=1, seed=7)
    assert len(output) == 2
    native, same = output
    assert native["decode_text"] == same["decode_text"]
    assert "### Incorrect" not in native["decode_text"]
    assert native["native_target_label"] != native["donor_target_label"]
    assert natural_schema_valid(native["native_target"])
    assert natural_schema_valid(native["donor_target"])
    assert native["native_control_role"] == "recipient_native"
    assert same["native_control_role"] == "same_rule_native"
    assert same["native_update_texts"] != native["native_update_texts"]
    assert receipt["same_question_query_identical"] is True


def test_offset_pair_ids_and_explicit_entity_exclusion_are_disjoint():
    corpus = []
    for index in range(8):
        corpus.extend(
            (
                row(f"{index}_A", "1", "adds instead of multiplying", "5"),
                row(f"{index}_B", "2", "subtracts instead of adding", "0"),
            )
        )
    train, _ = build_twins(corpus, twins=2, seed=7, offset=0)
    entities, bases = exclusion_sets(train)
    valid, receipt = build_twins(
        corpus,
        twins=1,
        seed=7,
        offset=2,
        excluded_entity_ids=entities,
        excluded_base_ids=bases,
    )
    fields = ("recipient_id", "donor_id", "same_rule_control_id")
    train_entities = {item[key] for item in train for key in fields}
    valid_entities = {item[key] for item in valid for key in fields}
    assert train_entities.isdisjoint(valid_entities)
    assert {item["same_question_base_id"] for item in train}.isdisjoint(
        {item["same_question_base_id"] for item in valid}
    )
    assert {item["pair_id"] for item in train}.isdisjoint(
        {item["pair_id"] for item in valid}
    )
    assert receipt["excluded_entity_ids"] == len(entities)
    assert receipt["excluded_base_ids"] == len(bases)
