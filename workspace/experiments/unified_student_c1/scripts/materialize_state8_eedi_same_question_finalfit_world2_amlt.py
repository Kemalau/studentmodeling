#!/usr/bin/env python3
"""Bind the exact full-epoch world2 job to the promoted State8 variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


VARIANTS = {
    "same_question_updater_mse1_seed3072",
    "same_question_full_mse1_seed3071",
    "same_question_repair_promoted",
    "same_question_repair_checkpoint_promoted",
    "same_question_curriculum_promoted",
}


def materialize(template: Path, promotion_path: Path, output: Path) -> None:
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    selected = promotion.get("selected") or {}
    if promotion.get("stage") != "state8_eedi_same_question_joint_promotion_v1":
        raise ValueError("Promotion stage drift")
    if promotion.get("pass") is not True or selected.get("qualified") is not True:
        raise ValueError("No jointly promoted candidate")
    variant = str(selected.get("finalfit_variant") or "")
    if variant not in VARIANTS:
        raise ValueError(f"Unexpected final-fit variant: {variant}")
    if selected.get("answer_C4_pass") is not True:
        raise ValueError("Answer-C4 did not pass")
    checks = selected.get("checks") or {}
    if checks.get("malrule_diagnosis_C4_pass") is not True:
        raise ValueError("MalRule diagnosis-C4 did not pass")
    if checks.get("eedi_same_question_C4_pass") is not True:
        raise ValueError("Eedi same-question diagnosis-C4 did not pass")
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Final-fit template job drift")
    env = jobs[0]["submit_args"]["env"]
    if env.get("FINALFIT_VARIANT") != "SELECTED_VARIANT":
        raise ValueError("Final-fit template was already bound")
    env["FINALFIT_VARIANT"] = variant
    jobs[0]["name"] = f"state8-eedi-{variant.replace('_', '-')}-unified-finalfit-seed42"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.template, args.promotion, args.output)
    print(f"materialized={args.output}")


if __name__ == "__main__":
    main()
