#!/usr/bin/env python3
"""Fail-closed validator for promoted-candidate formal evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from amlt_validation_utils import command_text

from validate_layerwise_unified_final_eval_amlt import (
    JOBS as BASE_JOBS,
    load,
    validate as validate_base,
)


PREFIX = "layerwise-promoted-final-eval-"
BASE_PREFIX = "layerwise-final-eval-"
RECEIPT = "layerwise_world_model_v1/task_retention_promotion/final_fit_receipt.json"


def base_name(name: str) -> str:
    if not name.startswith(PREFIX):
        raise ValueError(f"Promoted evaluation job drift: {name}")
    return BASE_PREFIX + name[len(PREFIX) :]


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
    jobs = config.get("jobs") or []
    observed = {str(job.get("name")) for job in jobs}
    expected = observed if expected_jobs is None else expected_jobs
    if observed != expected or not expected:
        raise ValueError("Promoted evaluation job-set drift")

    translated = dict(config)
    translated_jobs = []
    for job in jobs:
        item = dict(job)
        item["name"] = base_name(str(job.get("name")))
        translated_jobs.append(item)
    translated["jobs"] = translated_jobs
    translated_text_jobs = {base_name(name) for name in expected}
    if not translated_text_jobs.issubset(BASE_JOBS):
        raise ValueError("Promoted evaluation base job drift")

    # Reuse all target/GPU/image/protocol/runtime validation from the frozen
    # main evaluator after translating only the job names and receipt path.
    for job in translated_jobs:
        command = command_text(job.get("command"))
        command = command.replace(RECEIPT, "layerwise_world_model_v1/final_fit/seed42.json")
        command = command.replace(
            "x['status']=='final_fit_complete' and x['stage']=='layerwise_task_retention_promoted_final_fit'",
            "x['status']=='final_fit_complete' and x['stage']=='layerwise_unified_final_fit'",
        )
        command = command.replace("assert x['promoted_source_round']==750,x\n", "")
        command = command.replace(
            "c4_promoted_final_eval_progress_seed$${C4_PAIR_SEED}_v1",
            "c4_eval_progress_seed$${C4_PAIR_SEED}_v1",
        )
        command = command.replace(
            "c4_promoted_final_eval_progress_seed${C4_PAIR_SEED}_v1",
            "c4_eval_progress_seed${C4_PAIR_SEED}_v1",
        )
        command = command.replace(
            "stage=layerwise_task_retention_promoted_final_eval",
            "stage=layerwise_unified_final_eval",
        )
        job["command"] = command
    validate_base(translated, translated_text_jobs)

    for job in jobs:
        name = str(job.get("name"))
        command = command_text(job.get("command"))
        env = ((job.get("submit_args") or {}).get("env") or {})
        if env.get("VENV_ROOT") != "/tmp/studentmodeling-promoted-final-eval-venv":
            raise ValueError(f"Promoted evaluation venv drift: {name}")
        required = (
            RECEIPT,
            "layerwise_task_retention_promoted_final_fit",
            "promoted_source_round",
            "--state-architecture layerwise",
            "import accelerate,bitsandbytes,peft,safetensors,transformers",
        )
        if any(fragment not in command for fragment in required):
            raise ValueError(f"Promoted evaluation receipt/protocol drift: {name}")
        venv_functions = (
            'python() { "$VENV_ROOT/bin/python" "$@"; }',
            'python() { "$$VENV_ROOT/bin/python" "$$@"; }',
        )
        if not any(fragment in command for fragment in venv_functions):
            raise ValueError(f"Promoted evaluation venv launcher drift: {name}")
        if "-c4-seed" in name and not any(
            fragment in command
            for fragment in (
                "c4_promoted_final_eval_progress_seed$${C4_PAIR_SEED}_v1",
                "c4_promoted_final_eval_progress_seed${C4_PAIR_SEED}_v1",
            )
        ):
            raise ValueError("Promoted C4 progress namespace drift")
        if name.endswith("-t1-t6-seed42"):
            calibration_required = "apply_frozen_t6_threshold.py" in command
            receipt_forms = ('--receipt "$$RECEIPT"', '--receipt "$RECEIPT"')
            group_forms = (
                '"$$EVAL_GROUP" == t1_t6',
                '"$EVAL_GROUP" == t1_t6',
            )
            if (
                not calibration_required
                or not any(fragment in command for fragment in receipt_forms)
                or not any(fragment in command for fragment in group_forms)
            ):
                raise ValueError("Promoted T6 frozen-calibration drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
