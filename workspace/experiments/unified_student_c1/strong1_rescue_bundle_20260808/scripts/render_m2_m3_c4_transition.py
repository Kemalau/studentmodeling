#!/usr/bin/env python3
"""Render a frozen C4 transition audit across the M2 -> M3 hand-off.

The audit keeps the recurrent state and evaluator fixed while isolating the
adapter replacement performed at M3 initialization.  It is diagnostic only:
no optimizer step is executed and the frozen C4 manifest is unchanged.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_JOB = "m4-eval-c4-seed42"
CONDITIONS = {
    "m2_cf1_native": {
        "job": "m2-cf1-native-c4-seed42",
        "adapter": "/mnt/amulet/studentmodeling/checkpoints/recurrent_malrule_cf1_u4_trainable_seed42_v4_20260730/final_adapter",
        "state": "/mnt/amulet/studentmodeling/checkpoints/recurrent_malrule_cf1_u4_trainable_seed42_v4_20260730/recurrent_state.pt",
        "completion": "/mnt/amulet/studentmodeling/checkpoints/recurrent_malrule_cf1_u4_trainable_seed42_v4_20260730/completion.json",
    },
    "m2_state_full_adapter": {
        "job": "m2-state-full-adapter-c4-seed42",
        "adapter": "/mnt/amulet/studentmodeling/adapters/qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter",
        "state": "/mnt/amulet/studentmodeling/checkpoints/recurrent_malrule_cf1_u4_trainable_seed42_v4_20260730/recurrent_state.pt",
        "completion": "/mnt/amulet/studentmodeling/checkpoints/recurrent_malrule_cf1_u4_trainable_seed42_v4_20260730/completion.json",
    },
    "m3_full": {
        "job": "m3-generation-kd-c4-seed42",
        "adapter": "/mnt/amulet/studentmodeling/checkpoints/world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730/final_adapter",
        "state": "/mnt/amulet/studentmodeling/checkpoints/world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730/recurrent_state.pt",
        "completion": "/mnt/amulet/studentmodeling/checkpoints/world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730/completion.json",
    },
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def transition_command(source: str) -> str:
    start = source.index("export RECEIPT=")
    end = source.index('test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"')
    command = (
        source[:start]
        + 'export BASE_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean\n'
        + 'export ADAPTER_DIR="$${ADAPTER_DIR_REMOTE}"\n'
        + 'export STATE_PATH="$${STATE_PATH_REMOTE}"\n'
        + 'export SOURCE_COMPLETION="$${SOURCE_COMPLETION_REMOTE}"\n'
        + 'mkdir -p "$$HF_HUB_CACHE" "$$AMLT_OUTPUT_DIR/eval/c4"\n'
        + source[end:]
    )
    command = command.replace(
        'test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"',
        'test -f "$$ADAPTER_DIR/adapter_model.safetensors"',
        1,
    )
    command = command.replace(
        'test -f "$$RUN_DIR/recurrent_state.pt"',
        'test -f "$$STATE_PATH"',
        1,
    )
    command = command.replace(
        'test -f "$$BASE_DATA/test_same_template.jsonl"',
        'test -f "$$SOURCE_COMPLETION"\n'
        "python - \"$$SOURCE_COMPLETION\" <<'PY'\n"
        "import json, sys\n"
        "value = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "assert value.get('status') in {'complete', 'diagnostic_complete'}, value\n"
        "PY\n"
        'test -f "$$BASE_DATA/test_same_template.jsonl"',
        1,
    )
    command = command.replace(
        '--adapter "$$RUN_DIR/final_adapter"', '--adapter "$$ADAPTER_DIR"', 1
    )
    command = command.replace(
        '--recurrent-state "$$RUN_DIR/recurrent_state.pt"',
        '--recurrent-state "$$STATE_PATH"',
        1,
    )
    command += (
        "printf '%s\\n' \"condition=$$CONDITION\" "
        "\"adapter=$$ADAPTER_DIR\" \"state=$$STATE_PATH\" "
        "> \"$$AMLT_OUTPUT_DIR/eval/c4/transition_provenance.txt\"\n"
    )
    if "RECEIPT" in command or "RUN_DIR" in command:
        raise ValueError("Receipt-based model selection leaked into transition audit")
    return command


def render(base: dict) -> dict:
    jobs = base.get("jobs") or []
    source = next((job for job in jobs if job.get("name") == SOURCE_JOB), None)
    if source is None:
        raise ValueError(f"Source job {SOURCE_JOB!r} is missing")
    result = copy.deepcopy(base)
    result["description"] = (
        "Frozen 150-pair recurrent C4 transition audit: M2 CF1-native, M2 "
        "state with the full-task adapter, and the post-distillation M3 state"
    )
    result["jobs"] = []
    command = transition_command(str(source["command"]))
    for condition, spec in CONDITIONS.items():
        job = copy.deepcopy(source)
        job["name"] = spec["job"]
        job["sku"] = "80G1-A100"
        job["mpi"] = False
        job["process_count_per_node"] = 1
        job["command"] = command
        env = job.setdefault("submit_args", {}).setdefault("env", {})
        env.update(
            {
                "CONDITION": condition,
                "ADAPTER_DIR_REMOTE": spec["adapter"],
                "STATE_PATH_REMOTE": spec["state"],
                "SOURCE_COMPLETION_REMOTE": spec["completion"],
            }
        )
        result["jobs"].append(job)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(load(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output} jobs={len(value['jobs'])}")


if __name__ == "__main__":
    main()
