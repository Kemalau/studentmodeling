#!/usr/bin/env python3
"""Diagnosis-C4 evaluator with the source-aware taxonomy-label contract."""

from __future__ import annotations

import evaluate_recurrent_diagnosis_c4 as evaluator
from state8_taxonomy_contract import add_taxonomy_contract


ORIGINAL_PROMPT = evaluator.diagnosis_prompt


def taxonomy_prompt(row):
    value = dict(row)
    value.setdefault("source", "malrule")
    return add_taxonomy_contract(ORIGINAL_PROMPT(row), value)


if __name__ == "__main__":
    evaluator.diagnosis_prompt = taxonomy_prompt
    evaluator.main()

