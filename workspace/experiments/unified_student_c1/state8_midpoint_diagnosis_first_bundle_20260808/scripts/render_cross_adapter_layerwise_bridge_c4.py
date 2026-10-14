#!/usr/bin/env python3
"""Render a frozen factorial of CF1 state capture and task decoders."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from render_layerwise_state_bridge_c4 import bridge_command


SOURCE_JOB = "cf1-visionpad-layer-transport-singles-n50-seed42"
DECODERS = {
    "m3": (
        "/mnt/amulet/studentmodeling/checkpoints/"
        "world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730/"
        "final_adapter"
    ),
    "full_task": (
        "/mnt/amulet/studentmodeling/adapters/"
        "qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter"
    ),
}
MODES = ("compact", "one_shot_aligned")


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def cross_command(source: str) -> str:
    command = bridge_command(source)
    marker = 'test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"\n'
    if command.count(marker) != 1:
        raise ValueError("Could not locate capture-adapter preflight")
    command = command.replace(
        marker,
        marker + 'test -f "$$DECODE_ADAPTER/adapter_model.safetensors"\n',
        1,
    )
    argument = "  --adapter \"$$RUN_DIR/final_adapter\" \\\n"
    if command.count(argument) != 1:
        raise ValueError("Could not locate capture-adapter argument")
    command = command.replace(
        argument,
        argument + '  --decode-adapter "$$DECODE_ADAPTER" \\\n',
        1,
    )
    return command


def render(base: dict) -> dict:
    source = next(
        (job for job in base.get("jobs") or [] if job.get("name") == SOURCE_JOB),
        None,
    )
    if source is None:
        raise ValueError(f"Source job is missing: {SOURCE_JOB}")
    result = copy.deepcopy(base)
    result["description"] = (
        "Zero-training C4 factorial: capture the causal CF1 one-address "
        "layerwise state, then decode with M3 or the complete-task adapter"
    )
    command = cross_command(str(source["command"]))
    result["jobs"] = []
    for decoder, path in DECODERS.items():
        for mode in MODES:
            suffix = "aligned" if mode == "one_shot_aligned" else mode
            job = copy.deepcopy(source)
            job["name"] = f"cf1-state-{decoder}-decoder-{suffix}-n150-seed42"
            job["command"] = command
            job["sku"] = "80G1-A100"
            job["mpi"] = False
            job["process_count_per_node"] = 1
            env = job.setdefault("submit_args", {}).setdefault("env", {})
            env.clear()
            env.update(
                {
                    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
                    "POSITION_MODE": mode,
                    "DECODER": decoder,
                    "DECODE_ADAPTER": path,
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
