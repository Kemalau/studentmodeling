#!/usr/bin/env python3
"""Train-only open-generation view for Malrule T3 rows.

The original Malrule training prompt enumerates every valid misconception
target, while the frozen formal protocol intentionally requires open
generation.  This module removes only that train-time enumeration.  It does
not alter the target, recurrent updates, candidate metadata, direct-state
teacher prompt, formal evaluation rows, or C4 causal-pair rows.
"""

from __future__ import annotations

from typing import Any


VALID_OUTPUTS = "Valid outputs:\n"
QUERY_END = "</query>"


def is_open_t3_row(row: dict[str, Any]) -> bool:
    """Return whether a row receives the train-only open Malrule T3 view."""
    return (
        row.get("task") == "diagnose_misconception"
        and row.get("source") == "malrule"
    )


def selective_behavior_kd_weight(
    row: dict[str, Any], global_weight: float
) -> float:
    """Disable incompatible cached logits only for rewritten open-T3 rows."""
    if global_weight < 0:
        raise ValueError("Behavior KD weight must be non-negative")
    return 0.0 if is_open_t3_row(row) else float(global_weight)


def remove_valid_outputs(text: str) -> str:
    """Remove one candidate enumeration while preserving the query wrapper."""
    start = text.find(VALID_OUTPUTS)
    if start < 0:
        return text
    end = text.find(QUERY_END, start)
    if end < 0:
        raise ValueError("Malrule T3 prompt has Valid outputs without </query>")
    if text.find(VALID_OUTPUTS, start + len(VALID_OUTPUTS)) >= 0:
        raise ValueError("Malrule T3 prompt has multiple Valid outputs blocks")
    prefix = text[:start].rstrip()
    suffix = text[end:].lstrip()
    return f"{prefix}\n{suffix}"


def apply_open_t3_training_view(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Mutate only Malrule misconception decode prompts and return an audit."""
    eligible = changed = already_open = 0
    for row in rows:
        if not is_open_t3_row(row):
            continue
        eligible += 1
        before = str(row["decode_text"])
        after = remove_valid_outputs(before)
        if after == before:
            already_open += 1
            continue
        target = str(row["target"])
        if target in after:
            raise ValueError("Open T3 training prompt leaked its gold target")
        row["decode_text"] = after
        changed += 1
    return {
        "eligible_malrule_t3_rows": eligible,
        "changed_malrule_t3_rows": changed,
        "already_open_malrule_t3_rows": already_open,
    }
