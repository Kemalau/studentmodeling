#!/usr/bin/env python3
"""Render the frozen zero-training layerwise state-address bridge screen."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_JOB = "cf1-visionpad-layer-transport-singles-n50-seed42"
SCRIPT_SHA256 = "326042385d42111f96ff58195f1d421f2133adf9a20dbb5625c4d0d7b14004b3"
TEST_SHA256 = "4cd5ed86fad095ad1e9ad1736b11af92aeff4ea5b0ce1704799866d2fb56daf4"
SPECS = {
    "compact": "cf1-layerwise-state-bridge-compact-n150-seed42",
    "one_shot_aligned": "cf1-layerwise-state-bridge-aligned-n150-seed42",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def bridge_command(source: str) -> str:
    start = source.index(
        'test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"'
    )
    prefix = source[:start]
    return prefix + f'''test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"
test "$$(sha256sum scripts/evaluate_layerwise_state_bridge.py | cut -d' ' -f1)" = {SCRIPT_SHA256}
test "$$(sha256sum tests/test_layerwise_state_bridge.py | cut -d' ' -f1)" = {TEST_SHA256}
python -m unittest discover -s tests -p 'test_layerwise_state_bridge.py' -v
python -u scripts/evaluate_layerwise_state_bridge.py \\
  --model "$$LOCAL_MODEL" \\
  --adapter "$$RUN_DIR/final_adapter" \\
  --jsonl "$$C4_DATA/test_same_template.jsonl" \\
  --pair-manifest "$$PAIR_MANIFEST" \\
  --output-json "$$AMLT_OUTPUT_DIR/gate/layerwise_state_bridge.json" \\
  --state-token-alias '<|vision_pad|>' \\
  --position-mode "$$POSITION_MODE" \\
  --max-length 1536 \\
  --num-pairs 150 \\
  --seed 42
printf '%s\n' 'diagnostic_only=true' 'optimizer_steps=0' \\
  "position_mode=$$POSITION_MODE" \\
  'decoder_view=state_plus_current_query_strict_late_fusion' \\
  > "$$AMLT_OUTPUT_DIR/gate/scope.txt"
'''


def render(base: dict) -> dict:
    source = next(
        (job for job in base.get("jobs") or [] if job.get("name") == SOURCE_JOB),
        None,
    )
    if source is None:
        raise ValueError(f"Source job is missing: {SOURCE_JOB}")
    result = copy.deepcopy(base)
    result["description"] = (
        "Zero-training frozen C4 screen of one-address layerwise residual "
        "state transport into a strict history-free decoder"
    )
    command = bridge_command(str(source["command"]))
    result["jobs"] = []
    for mode, name in SPECS.items():
        job = copy.deepcopy(source)
        job["name"] = name
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
