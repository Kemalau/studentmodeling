#!/usr/bin/env python3
"""Render final MalRule C4 with diagnosis-first same-question lineage support."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "4f2ac16a767399ae2dd1ed6e08ed1af3086ff58b53a03ad6b4d8b2855f363832"


def replace(text: str, old: str, new: str, count: int = 1) -> str:
    observed = text.count(old)
    if observed != count:
        raise ValueError(f"replacement drift observed={observed} old={old!r}")
    return text.replace(old, new)


def render(source: Path, output: Path) -> str:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Final C4 source hash drift")
    text = raw.decode("utf-8")
    text = replace(
        text,
        'if [[ "$SELECTED_VARIANT" == strong1_rescue_taxonomy_joint_seed61 \\\n'
        '      || "$SELECTED_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \\\n'
        '      || "$SELECTED_VARIANT" == strong1_exact_* \\\n'
        '      || "$SELECTED_VARIANT" == diagnosis_first_matched_seed67 ]]; then',
        'if [[ "$SELECTED_VARIANT" == strong1_rescue_taxonomy_joint_seed61 \\\n'
        '      || "$SELECTED_VARIANT" == strong1_midpoint_diagnosis_first_seed63 \\\n'
        '      || "$SELECTED_VARIANT" == strong1_exact_* \\\n'
        '      || "$SELECTED_VARIANT" == diagnosis_first_matched_seed67 \\\n'
        '      || "$SELECTED_VARIANT" == same_question_* ]]; then',
    )
    diagnosis_set = "'diagnosis_first_matched_seed67'}"
    text = replace(
        text,
        diagnosis_set,
        "'diagnosis_first_matched_seed67','same_question_updater_mse1_seed3072','same_question_full_mse1_seed3071','same_question_repair_promoted','same_question_repair_checkpoint_promoted','same_question_curriculum_promoted'}",
        2,
    )
    text = replace(
        text,
        "if r['selected_replay_variant']=='diagnosis_first_matched_seed67'",
        "if r['selected_replay_variant']=='diagnosis_first_matched_seed67' or str(r['selected_replay_variant']).startswith('same_question_')",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.source, args.output))


if __name__ == "__main__":
    main()
