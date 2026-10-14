#!/usr/bin/env python3
"""Render the zero-training layerwise recurrent C4 factorial."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_JOB = "cf1-visionpad-layer-transport-singles-n50-seed42"
EVAL_SHA256 = "107e80057de4b477e3e19845ed875ae64e629876912cc4762b164af5fa763c1a"
MODEL_SHA256 = "b021e67f268c73445e7cf4488338b056d296364a57cfd67dcd4007ddde28fa20"
EVAL_TEST_SHA256 = "ae1de183704fa9beae536260d0f700d5b013138157c8f103a3deaf7283dc3c45"
MODEL_TEST_SHA256 = "165a931facdca974b3a73ef4b305a08bafd6c343bfb6f879d3d3be51f18bb26a"
CAPTURE_ADAPTER = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter"
)
DECODERS = {
    "cf1": CAPTURE_ADAPTER,
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


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def recurrent_command(source: str) -> str:
    start = source.index(
        'test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"'
    )
    prefix = source[:start]
    return prefix + f'''test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$$DECODE_ADAPTER/adapter_model.safetensors"
test "$$(sha256sum scripts/evaluate_layerwise_recurrent_causal_gate.py | cut -d' ' -f1)" = {EVAL_SHA256}
test "$$(sha256sum scripts/layerwise_recurrent_student_model.py | cut -d' ' -f1)" = {MODEL_SHA256}
test "$$(sha256sum tests/test_evaluate_layerwise_recurrent_causal_gate.py | cut -d' ' -f1)" = {EVAL_TEST_SHA256}
test "$$(sha256sum tests/test_layerwise_recurrent_student_model.py | cut -d' ' -f1)" = {MODEL_TEST_SHA256}
python -m unittest discover -s tests -p 'test_evaluate_layerwise_recurrent_causal_gate.py' -v
python -m unittest discover -s tests -p 'test_layerwise_recurrent_student_model.py' -v
python -u scripts/evaluate_layerwise_recurrent_causal_gate.py \\
  --model "$$LOCAL_MODEL" \\
  --adapter "$$RUN_DIR/final_adapter" \\
  --decode-adapter "$$DECODE_ADAPTER" \\
  --jsonl "$$C4_DATA/test_same_template.jsonl" \\
  --pair-manifest "$$PAIR_MANIFEST" \\
  --output-json "$$AMLT_OUTPUT_DIR/gate/layerwise_recurrent_c4.json" \\
  --state-token-alias '<|vision_pad|>' \\
  --max-length 1536 \\
  --num-pairs 150 \\
  --seed 42
printf '%s\n' 'diagnostic_only=true' 'optimizer_steps=0' \\
  "decoder=$$DECODER" \\
  'transition=one_address_layerwise_residual_recurrence' \\
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
        "Zero-training frozen C4 test of one-address layerwise residual "
        "recurrence with CF1, M3, and complete-task decoders"
    )
    command = recurrent_command(str(source["command"]))
    result["jobs"] = []
    for decoder, path in DECODERS.items():
        job = copy.deepcopy(source)
        job["name"] = f"layerwise-recurrent-{decoder}-decoder-n150-seed42"
        job["command"] = command
        job["sku"] = "80G1-A100"
        job["mpi"] = False
        job["process_count_per_node"] = 1
        env = job.setdefault("submit_args", {}).setdefault("env", {})
        env.clear()
        env.update(
            {
                "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
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
