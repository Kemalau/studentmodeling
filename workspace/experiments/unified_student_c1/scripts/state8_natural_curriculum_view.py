#!/usr/bin/env python3
"""Train-only natural-language T3 targets for the eight-slot curriculum."""

from __future__ import annotations

import ast
import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from open_t3_training_view import remove_valid_outputs
except ImportError:  # package-style unit tests
    from scripts.open_t3_training_view import remove_valid_outputs


MISCONCEPTION = re.compile(r"<misconception>\s*(.*?)\s*</misconception>", re.I | re.S)

# Code-side canonical supervision for the Malrule families used by the frozen
# Eedi-dense curriculum.  Keep this small resource beside the trainer because
# AMLT deliberately excludes the multi-GB data tree from code uploads.
CURRICULUM_DIAGNOSES = {
    "NONE": "No stable systematic misconception is supported by the observed interactions.",
    "decimals.ignore_decimal_point": "The student ignores the decimal point and treats decimal quantities as whole numbers.",
    "decimals.longer_is_larger": "The student assumes that a decimal with more digits is always larger.",
    "decimals.right_align_decimals": "The student right-aligns decimal digits instead of aligning the decimal points.",
    "decimals.whole_number_thinking": "The student applies whole-number comparison or arithmetic rules directly to decimals.",
    "fractions.add_numerators_denominators": "The student adds the numerators and denominators separately when adding fractions.",
    "fractions.common_denominator_numerator": "The student tries to create a common denominator by adding both numerators and denominators.",
    "fractions.denominator_comparison_error": "The student compares fractions by denominator size and incorrectly treats a larger denominator as a larger fraction.",
    "fractions.ignore_denominators": "The student operates only on fraction numerators and ignores the denominators.",
    "fractions.keep_common_denominator_for_multiplication": "The student keeps a shared denominator unchanged when multiplying fractions.",
    "fractions.multiply_across_for_division": "The student divides fractions by multiplying across without inverting the divisor.",
    "fractions.natural_number_bias_numerator_only": "The student compares fractions using only their numerators and ignores denominator effects.",
    "fractions.subtract_across": "The student subtracts fraction numerators and denominators component by component.",
    "subtraction.always_borrow_left": "The student borrows from the left column even when regrouping is unnecessary.",
    "subtraction.borrow_from_bottom": "When regrouping, the student decrements the bottom number instead of the top number.",
    "subtraction.borrow_no_decrement": "The student adds ten to a digit when borrowing but does not decrement the source digit.",
    "subtraction.decompose_by_place_value_label": "The student concatenates place-value labels instead of composing their numerical values.",
    "subtraction.diff_0_n_equals_n": "The student evaluates zero minus a positive digit as that positive digit instead of regrouping.",
    "subtraction.no_column_limit": "The student writes a multi-digit column result without carrying or respecting place-value columns.",
    "subtraction.smaller_from_larger": "The student subtracts the smaller digit from the larger digit in each column regardless of order.",
    "subtraction.stops_borrow_at_zero": "The student stops regrouping when a zero is encountered instead of borrowing across it.",
}


def diagnosis_value(target: str) -> str:
    match = MISCONCEPTION.fullmatch(str(target).strip())
    if not match:
        raise ValueError(f"T3 target lacks one misconception tag: {target!r}")
    return match.group(1).strip()


@lru_cache(maxsize=None)
def malrule_description(label: str) -> str | None:
    """Read the canonical Malrule definition without importing dataset code."""
    if label in CURRICULUM_DIAGNOSES:
        return CURRICULUM_DIAGNOSES[label]
    if not re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", label):
        return None
    family, rule = label.split(".", 1)
    path = (
        Path(__file__).resolve().parent.parent
        / "data/raw/malrulelib/malrules"
        / family
        / rule
        / "malrule_algorithm.py"
    )
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "get_description":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Return):
                continue
            try:
                value = ast.literal_eval(statement.value)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def humanized_label(label: str) -> str:
    """Fail open to natural language while keeping the canonical label separate."""
    if "." not in label:
        return label.replace("_", " ").strip()
    family, rule = label.split(".", 1)
    return (
        f"The student demonstrates {rule.replace('_', ' ')} when working with "
        f"{family.replace('_', ' ')}."
    )


def evidence_text(row: dict[str, Any]) -> str:
    updates = list(row.get("update_texts") or [])
    if not updates:
        return "The recorded student interaction is consistent with this error mechanism."
    observations: list[str] = []
    for update in updates[-2:]:
        match = re.search(
            r"<observed_interaction>(.*?)</observed_interaction>",
            str(update),
            flags=re.S,
        )
        if not match:
            continue
        try:
            observed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        student = observed.get("student_answer") or observed.get("student_response")
        correct = observed.get("correct_answer") or observed.get("correct_response")
        if student is not None and correct is not None:
            observations.append(
                f'The student answered "{str(student)[:120]}" instead of '
                f'"{str(correct)[:120]}".'
            )
            continue
        question = str(observed.get("question_text") or "")
        correct_match = re.search(r"###\s*Correct:\s*(.*?)(?:\n###|$)", question, re.S)
        wrong_match = re.search(r"###\s*Incorrect:\s*(.*?)(?:\n###|$)", question, re.S)
        if correct_match and wrong_match:
            observations.append(
                f'The student gave "{wrong_match.group(1).strip()[:120]}" instead of '
                f'"{correct_match.group(1).strip()[:120]}".'
            )
    return " ".join(observations) or (
        "The recorded wrong response, compared with the correct response, "
        "supports this specific error mechanism."
    )


def natural_target(row: dict[str, Any]) -> str:
    legacy_diagnosis = diagnosis_value(str(row["target"]))
    taxonomy_label = str(row.get("target_label") or legacy_diagnosis)
    label = str(row.get("semantic_label") or taxonomy_label)
    diagnosis = malrule_description(taxonomy_label) or (
        legacy_diagnosis
        if taxonomy_label != legacy_diagnosis
        or not re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", taxonomy_label)
        else humanized_label(taxonomy_label)
    )
    return (
        f"<label>{label}</label>\n"
        f"<diagnosis>{diagnosis}</diagnosis>\n"
        f"<evidence>{evidence_text(row)}</evidence>"
    )


def natural_prompt_text(text: str) -> str:
    text = remove_valid_outputs(str(text))
    patterns = (
        r"Return exactly one(?: valid)? (?:XML-style )?tag\.?",
        r"Return exactly one <misconception> tag\.?",
    )
    for pattern in patterns:
        text = re.sub(
            pattern,
            "Return exactly <label>, <diagnosis>, and <evidence> tags.",
            text,
            flags=re.I,
        )
    text = text.replace(
        "Return exactly one valid XML-style tag.",
        (
            "Return a canonical label plus a natural-language diagnosis and "
            "interaction-grounded evidence in <label>, <diagnosis>, and <evidence> tags."
        ),
    )
    # Some Malrule prompts carry a second, answer-adjacent legacy contract.
    # It overrides the earlier natural-schema instruction during generation
    # unless it is rewritten explicitly (the generic "Return exactly" rules
    # above do not match this line).
    text = re.sub(
        r"(?im)^\s*Output\s+format\s*:\s*<misconception>.*?</misconception>\s*$",
        "Output format: <label>canonical label</label> "
        "<diagnosis>natural-language diagnosis</diagnosis> "
        "<evidence>interaction-grounded evidence</evidence>",
        text,
    )
    return text


def apply_state8_natural_curriculum_view(rows: list[dict[str, Any]]) -> dict[str, int]:
    changed = 0
    by_source: dict[str, int] = {}
    for row in rows:
        if row.get("task") != "diagnose_misconception":
            continue
        row["target"] = natural_target(row)
        row["decode_text"] = natural_prompt_text(str(row["decode_text"]))
        if "direct_text" in row:
            row["direct_text"] = natural_prompt_text(str(row["direct_text"]))
        changed += 1
        source = str(row.get("source") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
    return {"changed_t3_rows": changed, **{f"source_{k}": v for k, v in sorted(by_source.items())}}
