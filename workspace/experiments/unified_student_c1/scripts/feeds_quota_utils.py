"""Robust parsing and selection of the fixed Feeds A100-80GB quota row."""

from __future__ import annotations

import json
from typing import Any


def parse_json_table(text: str) -> list[dict[str, Any]]:
    errors: list[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate.startswith("["):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as error:
            errors.append(str(error))
            continue
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return value
    raise ValueError(f"No JSON table found in AMLT output; parse_errors={errors[:3]}")


def select_a100_80gb(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("target_name") == "Feeds"
        and row.get("series") == "NDAMv4"
        and (row.get("accelerator") or {}).get("type") == "A100"
        and int((row.get("accelerator") or {}).get("memory_gb")) == 80
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Feeds NDAMv4/A100-80GB row, got {matches}")
    return matches[0]
