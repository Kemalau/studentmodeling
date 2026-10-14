#!/usr/bin/env python3
"""Diagnosis-first evaluation with query routing only for Eedi rows."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_recurrent_formal_protocols as evaluator
import state8_natural_curriculum_view as natural
from state8_current_query_routing import apply_current_query_routing
from state8_taxonomy_contract import apply_state8_taxonomy_contract
from train_state8_diagnosis_first_causal_repair import apply_diagnosis_first_order


ORIGINAL_APPLY = natural.apply_state8_natural_curriculum_view


def combined(rows):
    result = ORIGINAL_APPLY(rows)
    result.update(apply_state8_taxonomy_contract(rows))
    result.update(apply_diagnosis_first_order(rows))
    result.update(apply_current_query_routing(rows, sources={"eedi_mining"}))
    return result


if __name__ == "__main__":
    natural.apply_state8_natural_curriculum_view = combined
    evaluator.main()
