#!/usr/bin/env python3
"""Render a no-training layerwise transport audit for the causal CF1 adapter."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_JOB = "qwen3-32b-state-curriculum-stage-a-cf1-alias-gate"
SPECS = {
    "singles": {
        "job": "cf1-visionpad-layer-transport-singles-n50-seed42",
        "groups": "L56=56;L63=63",
    },
    "blocks": {
        "job": "cf1-visionpad-layer-transport-blocks-n50-seed42",
        "groups": "late8=56-63;prelate8=48-55",
    },
    "span": {
        "job": "cf1-visionpad-layer-transport-span-n50-seed42",
        "groups": "late16=48-63;all=0-63",
    },
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def render(base: dict) -> dict:
    source = next(
        (job for job in base.get("jobs") or [] if job.get("name") == SOURCE_JOB), None
    )
    if source is None:
        raise ValueError("Source Stage-A alias gate job is missing")
    result = copy.deepcopy(base)
    result["description"] = (
        "No-training frozen C4 layer-transport screen for the causal CF1 "
        "vision-pad adapter; 50 fixed pairs, grouped residual-stream patches"
    )
    command = str(source["command"])
    start = command.index('if [[ ! -f "$$RUN_DIR/final_adapter/adapter_model.safetensors" ]]; then')
    eval_start = command.index("python -u scripts/state_swap_misconception_smoke.py \\\n", start)
    eval_end = command.index("\n\npython -u scripts/summarize_stage_a_gate.py", eval_start)
    prefix = command[:start]
    eval_command = command[eval_start:eval_end]
    eval_command = eval_command.replace(
        '--output-json "$$AMLT_OUTPUT_DIR/gate/state_swap_masked.json"',
        '--output-json "$$AMLT_OUTPUT_DIR/gate/layer_transport_c4_n50.json"',
    )
    eval_command = eval_command.replace("--layers all", "--layers all \\\n        --layer-groups '$$LAYER_GROUPS'")
    eval_command = eval_command.replace("--num-pairs 150", "--num-pairs 50")
    eval_command = eval_command.replace(
        '--pair-manifest "$$PAIR_MANIFEST"',
        '--pair-manifest "$$PAIR_MANIFEST" \\\n        --allow-manifest-prefix',
    )
    command = (
        prefix
        + '      test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"\n'
        + '      python -m unittest discover -s tests -p \'test_state_swap_layer_groups.py\' -v\n'
        + eval_command
        + "\n      printf '%s\\n' 'diagnostic_only=true' 'frozen_prefix_n=50' "
          "\"layer_groups=$$LAYER_GROUPS\" > \"$$AMLT_OUTPUT_DIR/gate/scope.txt\"\n"
    )
    result["jobs"] = []
    for name, spec in SPECS.items():
        job = copy.deepcopy(source)
        job["name"] = spec["job"]
        job["command"] = command
        job["sku"] = "80G1-A100"
        job["mpi"] = False
        job["process_count_per_node"] = 1
        env = job.setdefault("submit_args", {}).setdefault("env", {})
        env.clear()
        env.update(
            {
                "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
                "LAYER_GROUPS": spec["groups"],
                "TRANSPORT_GROUP": name,
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
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output}")


if __name__ == "__main__":
    main()
