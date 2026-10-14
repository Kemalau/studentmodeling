#!/usr/bin/env python3
"""State8 taxonomy repair with semantic fields before the auxiliary label.

The numeric Eedi taxonomy has more than one thousand classes.  Emitting its
label first can trap an early student checkpoint in a repeated-digit prefix,
preventing the decoder from ever reaching the semantically authoritative
diagnosis and evidence fields.  This entrypoint keeps the same strict
state-only teacher, taxonomy contract, causal objectives, and eight-slot
state, but trains the order

    <diagnosis>...</diagnosis><evidence>...</evidence><label>...</label>

and gives the canonical label ordinary (1x) token weight.  The label remains
present as auxiliary supervision and reporting metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_semantic_causal_repair as semantic
import train_state8_taxonomy_causal_repair as taxonomy


FIELD_ORDER = ("diagnosis", "evidence", "label")
TAG = {
    name: re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.I | re.S)
    for name in ("label", "diagnosis", "evidence")
}


def reorder_target(value: str) -> str:
    """Move the auxiliary label behind the two semantic fields."""
    text = str(value)
    matches = {name: pattern.findall(text) for name, pattern in TAG.items()}
    present = {name for name, values in matches.items() if values}
    if not present:
        return text
    if present != set(TAG) or any(len(values) != 1 for values in matches.values()):
        raise ValueError("Diagnosis target must contain exactly one label/diagnosis/evidence triple")
    return "\n".join(f"<{name}>{matches[name][0].strip()}</{name}>" for name in FIELD_ORDER)


def diagnosis_first_prompt(value: str) -> str:
    """Make the requested order explicit without exposing any gold label."""
    text = str(value)
    replacements = (
        (
            r"Return exactly one non-empty <label>, <diagnosis>, and <evidence> tag\.",
            "Return exactly one non-empty <diagnosis>, <evidence>, and <label> tag, "
            "in that order. The natural-language diagnosis is primary and the "
            "canonical label is auxiliary.",
        ),
        (
            r"Return exactly <label>, <diagnosis>, and <evidence> tags\.",
            "Return exactly <diagnosis>, <evidence>, and <label> tags, in that order. "
            "The natural-language diagnosis is primary and the canonical label is auxiliary.",
        ),
        (
            r"Return a canonical label plus a natural-language diagnosis and interaction-grounded "
            r"evidence in <label>, <diagnosis>, and <evidence> tags\.",
            "Return a natural-language diagnosis and interaction-grounded evidence first, "
            "followed by an auxiliary canonical label, in <diagnosis>, <evidence>, and <label> tags.",
        ),
        (
            r"Output format:\s*<label>canonical label</label>\s*"
            r"<diagnosis>natural-language diagnosis</diagnosis>\s*"
            r"<evidence>interaction-grounded evidence</evidence>",
            "Output format: <diagnosis>natural-language diagnosis</diagnosis> "
            "<evidence>interaction-grounded evidence</evidence> "
            "<label>auxiliary canonical label</label>",
        ),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def apply_diagnosis_first_order(rows: list[dict[str, Any]]) -> dict[str, int]:
    target_rows = 0
    prompt_rows = 0
    for row in rows:
        is_diagnosis = (
            str(row.get("task") or "") == "diagnose_misconception"
            or str(row.get("causal_target_kind") or "") == "diagnosis_natural_schema"
        )
        if not is_diagnosis:
            continue
        changed_target = False
        for key in ("target", "native_target", "donor_target"):
            if key not in row:
                continue
            before = str(row[key])
            after = reorder_target(before)
            row[key] = after
            changed_target |= after != before
        if isinstance(row.get("candidate_targets"), dict):
            before = dict(row["candidate_targets"])
            after = {key: reorder_target(str(value)) for key, value in before.items()}
            row["candidate_targets"] = after
            changed_target |= after != before
        changed_prompt = False
        for key in (
            "decode_text", "direct_text", "full_context_text", "prompt",
            "native_direct_text", "donor_direct_text",
        ):
            if key not in row:
                continue
            before = str(row[key])
            after = diagnosis_first_prompt(before)
            row[key] = after
            changed_prompt |= after != before
        target_rows += int(changed_target)
        prompt_rows += int(changed_prompt)
    return {
        "diagnosis_first_target_rows": target_rows,
        "diagnosis_first_prompt_rows": prompt_rows,
    }


def main() -> None:
    output_dir: Path | None = None
    audits: list[dict[str, Any]] = []
    original_parse = argparse.ArgumentParser.parse_args
    original_read = trainer.read_jsonl
    original_natural = taxonomy.apply_state8_natural_curriculum_view
    original_neutral = semantic.NEUTRAL_TARGET

    def parse(parser, *args, **kwargs):
        nonlocal output_dir
        namespace = original_parse(parser, *args, **kwargs)
        output_dir = Path(namespace.output_dir)
        namespace.diagnosis_field_order = list(FIELD_ORDER)
        namespace.canonical_label_role = "auxiliary_ordinary_weight"
        return namespace

    def read(path):
        rows = original_read(path)
        audit = apply_diagnosis_first_order(rows)
        if sum(audit.values()):
            audits.append({"scope": "loaded_rows", "path": str(path), **audit})
        return rows

    def natural_view(rows):
        audit = dict(original_natural(rows))
        order = apply_diagnosis_first_order(rows)
        audits.append({"scope": "natural_view", **order})
        audit.update(order)
        return audit

    argparse.ArgumentParser.parse_args = parse
    trainer.read_jsonl = read
    taxonomy.apply_state8_natural_curriculum_view = natural_view
    semantic.NEUTRAL_TARGET = reorder_target(original_neutral)
    try:
        taxonomy.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if output_dir is None:
                raise RuntimeError("Diagnosis-first repair did not observe output directory")
            target_rows = sum(int(item["diagnosis_first_target_rows"]) for item in audits)
            prompt_rows = sum(int(item["diagnosis_first_prompt_rows"]) for item in audits)
            if target_rows <= 0 or prompt_rows <= 0:
                raise RuntimeError("Diagnosis-first repair changed no supervised targets or prompts")
            receipt = {
                "stage": "state8_diagnosis_first_causal_repair_v1",
                "status": "complete",
                "state_token_count": 8,
                "field_order": list(FIELD_ORDER),
                "semantic_fields_primary": True,
                "canonical_label_role": "auxiliary_ordinary_weight",
                "label_auxiliary_weight": 1.0,
                "raw_history_visible_to_decoder": False,
                "final_inference_route": "strict_state_only",
                "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audits": audits,
            }
            (output_dir / "diagnosis_first_causal_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.read_jsonl = original_read
        taxonomy.apply_state8_natural_curriculum_view = original_natural
        semantic.NEUTRAL_TARGET = original_neutral


if __name__ == "__main__":
    main()
