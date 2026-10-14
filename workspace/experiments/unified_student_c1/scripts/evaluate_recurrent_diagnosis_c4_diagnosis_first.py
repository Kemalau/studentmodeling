#!/usr/bin/env python3
"""Diagnosis-C4 evaluator matching the diagnosis-first training contract."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_recurrent_diagnosis_c4 as evaluator
from state8_taxonomy_contract import add_taxonomy_contract
from train_state8_diagnosis_first_causal_repair import (
    diagnosis_first_prompt,
    reorder_target,
)


ORIGINAL_PROMPT = evaluator.diagnosis_prompt
ORIGINAL_TARGET = evaluator.diagnosis_target


def prompt(row):
    value = dict(row)
    # This evaluator is defined only for the frozen Malrule C4 panel.  The
    # source rows also contain a numeric ``target_label`` for the *response*
    # task; allowing the generic taxonomy helper to inspect that field makes
    # it look like an Eedi misconception ID.  Bind the prompt contract to the
    # known evaluation source instead of inferring it from answer metadata.
    value["source"] = "malrule"
    for key in (
        "target_label",
        "native_target_label",
        "recipient_label",
        "donor_target_label",
        "donor_label",
        "target",
        "native_target",
        "donor_target",
    ):
        value.pop(key, None)
    return diagnosis_first_prompt(add_taxonomy_contract(ORIGINAL_PROMPT(row), value))


def target(label):
    """Keep judge references in the same semantic-first order as training."""
    return reorder_target(ORIGINAL_TARGET(label))


if __name__ == "__main__":
    evaluator.diagnosis_prompt = prompt
    evaluator.diagnosis_target = target
    evaluator.main()
