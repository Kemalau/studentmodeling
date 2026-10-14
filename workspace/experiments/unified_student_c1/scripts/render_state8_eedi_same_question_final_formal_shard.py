#!/usr/bin/env python3
"""Render the final formal shard with same-question Eedi routing."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "f946ad13a560f59fde1869d6987ae25c00d48c92794477b4148d118b4b9fea49"


def replace(text: str, old: str, new: str, count: int = 1) -> str:
    observed = text.count(old)
    if observed != count:
        raise ValueError(f"replacement drift observed={observed} old={old!r}")
    return text.replace(old, new)


def render(source: Path, output: Path) -> str:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Final formal source hash drift")
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
    old = '''  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py
  test -f "$RUN_DIR/diagnosis_first_causal_repair.json"
  test "$(sha256sum "$FORMAL_EVALUATOR" | cut -d' ' -f1)" = a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
fi
'''
    new = old + '''if [[ "$SELECTED_VARIANT" == same_question_* ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py
  test -f "$RUN_DIR/eedi_same_question_finalfit.json"
  test "$(sha256sum "$FORMAL_EVALUATOR" | cut -d' ' -f1)" = 967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f
fi
'''
    text = replace(text, old, new)
    diagnosis_set = "'diagnosis_first_matched_seed67'}"
    text = replace(
        text,
        diagnosis_set,
        "'diagnosis_first_matched_seed67','same_question_updater_mse1_seed3072','same_question_full_mse1_seed3071','same_question_repair_promoted','same_question_repair_checkpoint_promoted','same_question_curriculum_promoted'}",
        2,
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
