#!/usr/bin/env python3
"""Unambiguous label contracts for natural-language State8 diagnosis.

The natural schema deliberately keeps a canonical label as auxiliary
supervision, but ``<label>canonical label</label>`` is ambiguous next to a math
query: a decoder can reasonably interpret it as the answer label.  This module
adds a source-aware *syntax* contract without exposing any candidate labels or
the gold class at inference time.
"""

from __future__ import annotations

import re
from typing import Any


CONTRACT_OPEN = "<misconception_label_contract>"
CONTRACT_CLOSE = "</misconception_label_contract>"
CANONICAL_ID = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
INTEGER_ID = re.compile(r"^[0-9]+$")
LABEL = re.compile(r"<(?:label|misconception)>\s*(.*?)\s*</(?:label|misconception)>", re.I | re.S)


def canonical_label(row: dict[str, Any]) -> str:
    for key in (
        "target_label",
        "native_target_label",
        "recipient_label",
        "donor_target_label",
        "donor_label",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    for key in ("target", "native_target", "donor_target"):
        match = LABEL.search(str(row.get(key) or ""))
        if match:
            return match.group(1).strip()
    return ""


def taxonomy_kind(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "").casefold()
    label = canonical_label(row)
    if source == "eedi_mining" or INTEGER_ID.fullmatch(label):
        return "eedi_id"
    if source == "malrule" or label == "NONE" or CANONICAL_ID.fullmatch(label):
        return "malrule_id"
    return "generic_id"


def label_contract(row: dict[str, Any]) -> str:
    kind = taxonomy_kind(row)
    if kind == "malrule_id":
        syntax = (
            "For this Malrule diagnosis, use a lowercase taxonomy identifier of "
            "the form domain.rule_name (letters, digits, underscores, and exactly "
            "one domain separator), or NONE when no stable misconception exists."
        )
    elif kind == "eedi_id":
        syntax = (
            "For this Eedi diagnosis, use the canonical numeric misconception ID "
            "from the Eedi taxonomy. A number is valid here only as a taxonomy ID, "
            "never as the answer to the math problem."
        )
    else:
        syntax = "Use the canonical misconception identifier from the task taxonomy."
    return (
        f"{CONTRACT_OPEN}\n"
        "The <label> field names the student's misconception class; it is not a "
        "prediction of the current problem's answer, score, or correctness. "
        f"{syntax} Do not solve the query or copy a mathematical answer into "
        "<label>. The <diagnosis> and <evidence> fields must describe and support "
        "that same misconception.\n"
        f"{CONTRACT_CLOSE}"
    )


def add_taxonomy_contract(text: str, row: dict[str, Any]) -> str:
    value = str(text)
    if CONTRACT_OPEN in value or CONTRACT_CLOSE in value:
        if value.count(CONTRACT_OPEN) != 1 or value.count(CONTRACT_CLOSE) != 1:
            raise ValueError("Malformed or duplicated misconception-label contract")
        return value
    contract = label_contract(row)
    anchors = ("<STUDENT_STATE>", "<student_history>", "<query>", "Answer:")
    for anchor in anchors:
        if anchor in value:
            return value.replace(anchor, f"{contract}\n{anchor}", 1)
    return f"{contract}\n{value}"


def apply_state8_taxonomy_contract(rows: list[dict[str, Any]]) -> dict[str, int]:
    changed = 0
    by_kind: dict[str, int] = {}
    for row in rows:
        task = str(row.get("task") or "")
        causal_kind = str(row.get("causal_target_kind") or "")
        if task != "diagnose_misconception" and causal_kind != "diagnosis_natural_schema":
            continue
        row_changed = False
        for key in ("decode_text", "direct_text", "prompt", "full_context_text"):
            if key not in row:
                continue
            before = str(row[key])
            after = add_taxonomy_contract(before, row)
            row[key] = after
            row_changed |= after != before
        if not any(key in row for key in ("decode_text", "direct_text", "prompt", "full_context_text")):
            raise ValueError(f"Diagnosis row lacks a prompt field: {row.get('example_id')}")
        if row_changed:
            changed += 1
        kind = taxonomy_kind(row)
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {
        "taxonomy_contract_rows": sum(by_kind.values()),
        "taxonomy_contract_changed_rows": changed,
        **{f"taxonomy_kind_{key}": value for key, value in sorted(by_kind.items())},
    }

