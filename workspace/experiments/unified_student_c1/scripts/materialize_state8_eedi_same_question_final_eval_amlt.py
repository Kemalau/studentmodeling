#!/usr/bin/env python3
"""Derive the 20-job final evaluator for the same-question canonical model."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import yaml


CANONICAL = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_eedi_same_question_unified_canonical_v1_20260811"
)
PROGRESS = "state8_eedi_same_question_final_full259845"


def materialize(source: Path, output: Path) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    jobs = config.get("jobs") or []
    if len(jobs) != 19:
        raise ValueError(f"Expected 19 base jobs, observed {len(jobs)}")
    c4_template = None
    for job in jobs:
        env = (job.setdefault("submit_args", {})).setdefault("env", {})
        env["STATE8_FINAL_EVAL_RUN_DIR"] = CANONICAL
        env["STATE8_FINAL_EVAL_PROGRESS_NAMESPACE"] = PROGRESS
        name = str(job.get("name") or "")
        command = str(job.get("command") or "")
        if name.startswith("state8-final-formal-shard"):
            old = "bash scripts/run_state8_final_formal_shard.sh"
            if command.count(old) != 1:
                raise ValueError(f"Formal source launcher drift: {name}")
            job["command"] = command.replace(
                old, "bash scripts/run_state8_eedi_same_question_final_formal_shard.sh"
            )
        elif name.startswith("state8-final-c4-seed"):
            c4_template = c4_template or copy.deepcopy(job)
            old = "bash scripts/run_state8_final_c4.sh"
            if command.count(old) != 1:
                raise ValueError(f"C4 source launcher drift: {name}")
            job["command"] = command.replace(
                old, "bash scripts/run_state8_eedi_same_question_final_malrule_c4.sh"
            )
        else:
            raise ValueError(f"Unexpected base job: {name}")
    if c4_template is None:
        raise ValueError("No C4 template job")
    eedi_job = copy.deepcopy(c4_template)
    eedi_job["name"] = "state8-final-eedi-same-question-c4-final150"
    eedi_job["command"] = str(eedi_job["command"]).replace(
        "bash scripts/run_state8_final_c4.sh",
        "bash scripts/run_state8_eedi_same_question_final_eedi_c4.sh",
    )
    env = eedi_job["submit_args"]["env"]
    env.clear()
    env.update(
        _AZUREML_SINGULARITY_JOB_UAI="rankfun_aml",
        VENV_ROOT="/tmp/studentmodeling-state8-final-eval-venv",
        STATE8_FINAL_EVAL_RUN_DIR=CANONICAL,
        STATE8_FINAL_EVAL_PROGRESS_NAMESPACE=PROGRESS,
    )
    jobs.append(eedi_job)
    config["description"] = (
        "Same-question State8 exact-epoch final evaluation: 16 formal shards, "
        "three MalRule answer/diagnosis C4 seeds, and one frozen 150-pair Eedi C4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
