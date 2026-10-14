#!/usr/bin/env python3
"""Render an M2-causal-seeded counterpart to the existing CF4 screen.

The data, optimizer horizon, objective, full-task teacher cache, and frozen C4
gate are identical to the single-A100 CF4 hedge.  Only the initialization is
changed from post-M3 artifacts to the complete M2 causal adapter plus state.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_JOB = "m4-causal-cf4-donor-directional-w1-single-gb4-step0to1000-seed42"
JOB = "m2-seeded-cf4-donor-directional-single-gb4-step0to1000-seed42"
BRANCH = "m2_seeded_donor_directional_cf4_w1_single_gb4"
M2_RUN_DIR = "/mnt/amulet/studentmodeling/checkpoints/recurrent_malrule_cf1_u4_trainable_seed42_v4_20260730"
RUN_DIR = "/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m2_seeded_cf4_donor_directional_w1_seed42_step1000_20260731"
OLD_TRAINER_SHA256 = "865c12219deb23cf9db145c797244969721fce80323050dcd987b33d96cbfb1b"
INTERMEDIATE_TRAINER_SHA256 = "338bf4270bb131a31bb4ba500cd9bc7229718a7caf23f5f1faa8f705f5ac8260"
TRAINER_SHA256 = "ab25b40f2b4e91582c9a3bd8bd784161f44aaa6ff5c7aa4c6d19da81693bac32"
OLD_EVALUATOR_SHA256 = "9dca3271d29153baf66b440239467e6889c0f2c64677c4e31fd4e2d8a5710c10"
EVALUATOR_SHA256 = "d2a0cf7702f3aaf56bc3629af7b63837f739799f47a3b51e5f5076917116f0b2"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def replace_all(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Missing source fragment: {old!r}")
    return text.replace(old, new)


def render(base: dict) -> dict:
    source = next(
        (job for job in base.get("jobs") or [] if job.get("name") == SOURCE_JOB),
        None,
    )
    if source is None:
        raise ValueError("Source single-A100 CF4 job is missing")
    result = copy.deepcopy(base)
    result["description"] = (
        "M2-causal-seeded CF4 screen matched to the post-M3 single-A100 hedge; "
        "only adapter/state initialization differs"
    )
    job = copy.deepcopy(source)
    job["name"] = JOB
    command = str(job["command"])
    source_hash = (
        INTERMEDIATE_TRAINER_SHA256
        if INTERMEDIATE_TRAINER_SHA256 in command
        else OLD_TRAINER_SHA256
    )
    command = replace_all(command, source_hash, TRAINER_SHA256)
    command = replace_all(command, OLD_EVALUATOR_SHA256, EVALUATOR_SHA256)
    command = replace_all(
        command,
        "export M3_RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730",
        f"export INIT_RUN_DIR={M2_RUN_DIR}",
    )
    command = replace_all(command, '"$$M3_RUN_DIR/final_adapter', '"$$INIT_RUN_DIR/final_adapter')
    command = replace_all(command, '"$$M3_RUN_DIR/recurrent_state.pt', '"$$INIT_RUN_DIR/recurrent_state.pt')
    command = replace_all(
        command,
        'export RUN_DIR="/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m4_causal_cf4_$${BRANCH}_seed42_step1000_20260731"',
        f'export RUN_DIR="{RUN_DIR}"',
    )
    command = replace_all(
        command,
        "assert branch == 'donor_directional_cf4_w1_single_gb4', branch",
        f"assert branch == '{BRANCH}', branch",
    )
    # Persist explicit initialization provenance in both the run config and the
    # AMLT result.  This job must never silently fall back to post-M3 artifacts.
    marker = 'test -f "$$INIT_RUN_DIR/recurrent_state.pt"\n'
    command = replace_all(
        command,
        marker,
        marker
        + 'test -f "$$INIT_RUN_DIR/completion.json"\n'
        + "python - \"$$INIT_RUN_DIR/completion.json\" <<'PY'\n"
        + "import json, sys\n"
        + "value = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        + "assert value.get('status') == 'complete', value\n"
        + "PY\n",
    )
    command += (
        "printf '%s\\n' 'initialization=m2_cf1_causal_adapter_and_state' "
        "\"init_run_dir=$$INIT_RUN_DIR\" > \"$$AMLT_OUTPUT_DIR/metrics/init_provenance.txt\"\n"
    )
    job["command"] = command
    env = job.setdefault("submit_args", {}).setdefault("env", {})
    env["BRANCH"] = BRANCH
    result["jobs"] = [job]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(load(args.base))
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output}")


if __name__ == "__main__":
    main()
