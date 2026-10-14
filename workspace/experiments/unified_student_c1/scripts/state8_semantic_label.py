#!/usr/bin/env python3
"""Readable, provenance-preserving labels for State8 diagnosis supervision.

Eedi's public taxonomy uses arbitrary integer identifiers.  Those identifiers
are useful for dataset joins, but they are a poor train-only readout target for
a latent student state: the digits contain no information about the associated
misconception.  This module derives a deterministic semantic slug from the
gold diagnosis while retaining the original integer as an explicit suffix and
as a separate ``taxonomy_id`` field.

The semantic label is auxiliary supervision.  Natural-language semantic
judgment remains authoritative and no label or candidate list is exposed to
the inference-time decoder.
"""

from __future__ import annotations

import re
from typing import Any


LABEL = re.compile(r"<label>\s*(.*?)\s*</label>", re.I | re.S)
DIAGNOSIS = re.compile(r"<diagnosis>\s*(.*?)\s*</diagnosis>", re.I | re.S)
MISCONCEPTION = re.compile(
    r"<misconception>\s*(.*?)\s*</misconception>", re.I | re.S
)
MALRULE_ID = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
EEDI_ID = re.compile(r"^[0-9]+$")
SEMANTIC_EEDI = re.compile(r"^eedi\.[a-z0-9_]+_id_[0-9]+$")


def _tag_value(text: Any, patterns: tuple[re.Pattern[str], ...]) -> str:
    value = str(text or "")
    for pattern in patterns:
        match = pattern.search(value)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return ""


def diagnosis_text(value: Any) -> str:
    """Extract the semantic diagnosis from legacy or natural-schema targets."""
    text = _tag_value(value, (DIAGNOSIS, MISCONCEPTION))
    if not text:
        raise ValueError("Diagnosis target has neither <diagnosis> nor <misconception>")
    return text


def _slug(text: str, *, max_characters: int = 88) -> str:
    words = re.findall(r"[a-z0-9]+", str(text).casefold())
    if not words:
        raise ValueError(f"Cannot derive semantic label from {text!r}")
    slug = "_".join(words)
    if len(slug) > max_characters:
        slug = slug[:max_characters].rstrip("_")
    if not slug:
        raise ValueError(f"Semantic label slug became empty for {text!r}")
    return slug


def eedi_semantic_label(taxonomy_id: Any, diagnosis: Any) -> str:
    """Return a readable Eedi label that remains uniquely taxonomy-bound."""
    identifier = str(taxonomy_id or "").strip()
    if not EEDI_ID.fullmatch(identifier):
        raise ValueError(f"Invalid Eedi taxonomy ID: {taxonomy_id!r}")
    value = f"eedi.{_slug(str(diagnosis))}_id_{identifier}"
    if not SEMANTIC_EEDI.fullmatch(value):
        raise AssertionError(f"Malformed Eedi semantic label: {value}")
    return value


def semantic_label(
    row: dict[str, Any],
    *,
    label_key: str = "target_label",
    target_key: str = "target",
) -> str:
    """Resolve the auxiliary label without changing the source taxonomy ID."""
    existing = str(row.get("semantic_label") or "").strip()
    if existing:
        if not (SEMANTIC_EEDI.fullmatch(existing) or MALRULE_ID.fullmatch(existing)):
            raise ValueError(f"Invalid existing semantic label: {existing!r}")
        return existing
    label = str(row.get(label_key) or "").strip()
    source = str(row.get("source") or "").casefold()
    if source == "eedi_mining" or EEDI_ID.fullmatch(label):
        return eedi_semantic_label(label, diagnosis_text(row.get(target_key)))
    if label == "NONE" or MALRULE_ID.fullmatch(label):
        return label
    if not label:
        return _slug(diagnosis_text(row.get(target_key)))
    raise ValueError(f"Unsupported diagnosis label contract: source={source!r} label={label!r}")


def replace_natural_label(target: Any, label: str) -> str:
    """Replace only the auxiliary label in a valid natural-schema target."""
    value = str(target)
    matches = list(LABEL.finditer(value))
    if len(matches) != 1:
        raise ValueError("Natural target must contain exactly one <label> field")
    return LABEL.sub(f"<label>{label}</label>", value, count=1)

