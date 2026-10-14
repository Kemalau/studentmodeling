from __future__ import annotations

from scripts.state8_taxonomy_contract import (
    CONTRACT_OPEN,
    add_taxonomy_contract,
    apply_state8_taxonomy_contract,
    taxonomy_kind,
)


def test_malrule_contract_forbids_answer_labels_and_specifies_id_syntax() -> None:
    row = {
        "task": "diagnose_misconception",
        "source": "malrule",
        "target_label": "fractions.denominator_comparison_error",
        "decode_text": "Diagnose.\n<STUDENT_STATE>\n<query>1/2 or 1/3?</query>\nAnswer:",
    }
    audit = apply_state8_taxonomy_contract([row])
    assert audit["taxonomy_kind_malrule_id"] == 1
    assert "domain.rule_name" in row["decode_text"]
    assert "not a prediction of the current problem's answer" in row["decode_text"]
    assert "fractions.denominator_comparison_error" not in row["decode_text"]
    assert row["decode_text"].count(CONTRACT_OPEN) == 1


def test_eedi_contract_allows_only_numeric_taxonomy_id_not_math_answer() -> None:
    row = {
        "task": "diagnose_misconception",
        "source": "eedi_mining",
        "target_label": "1990",
        "direct_text": "Diagnose.\n<student_history>h</student_history>\nAnswer:",
    }
    apply_state8_taxonomy_contract([row])
    assert taxonomy_kind(row) == "eedi_id"
    assert "numeric misconception ID" in row["direct_text"]
    assert "never as the answer to the math problem" in row["direct_text"]
    assert "1990" not in row["direct_text"]


def test_contract_is_idempotent_and_does_not_expose_candidates() -> None:
    row = {
        "causal_target_kind": "diagnosis_natural_schema",
        "recipient_label": "subtraction.borrow_no_decrement",
        "donor_label": "subtraction.smaller_from_larger",
        "decode_text": "<STUDENT_STATE>\n<query>q</query>\nAnswer:",
    }
    first = add_taxonomy_contract(row["decode_text"], row)
    second = add_taxonomy_contract(first, row)
    assert first == second
    assert row["recipient_label"] not in first
    assert row["donor_label"] not in first


def test_taxonomy_evaluator_wrappers_bind_the_new_contract() -> None:
    diagnosis = open("scripts/evaluate_recurrent_diagnosis_c4_taxonomy.py", encoding="utf-8").read()
    formal = open("scripts/evaluate_recurrent_formal_protocols_taxonomy.py", encoding="utf-8").read()
    trainer = open("scripts/train_state8_taxonomy_causal_repair.py", encoding="utf-8").read()
    assert "add_taxonomy_contract" in diagnosis
    assert "apply_state8_taxonomy_contract" in formal
    assert "candidate_taxonomy_visible_to_decoder" in trainer

