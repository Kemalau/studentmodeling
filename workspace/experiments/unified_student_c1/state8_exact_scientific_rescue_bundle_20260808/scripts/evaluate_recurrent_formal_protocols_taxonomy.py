#!/usr/bin/env python3
"""Closed-state formal evaluator using the explicit taxonomy-label contract."""

from __future__ import annotations

import evaluate_recurrent_formal_protocols as evaluator
import state8_natural_curriculum_view as natural
from state8_taxonomy_contract import apply_state8_taxonomy_contract


ORIGINAL_APPLY = natural.apply_state8_natural_curriculum_view


def combined(rows):
    result = ORIGINAL_APPLY(rows)
    result.update(apply_state8_taxonomy_contract(rows))
    return result


if __name__ == "__main__":
    natural.apply_state8_natural_curriculum_view = combined
    evaluator.main()

