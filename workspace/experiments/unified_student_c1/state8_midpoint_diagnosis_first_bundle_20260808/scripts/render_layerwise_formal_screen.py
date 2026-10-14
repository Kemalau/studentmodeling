#!/usr/bin/env python3
"""Render a four-job formal screen for the zero-training layerwise state."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


CAPTURE_ADAPTER = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter"
)
DECODE_ADAPTER = (
    "/mnt/amulet/studentmodeling/adapters/"
    "qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter"
)
EVALUATOR_SHA256 = "eb0243cf232d7e634f0ad784d30e66f93f42f84833a748ec453b648cff0eccda"
LAYERWISE_SHA256 = "b6e63f4321962e8569c9c7955ba9620a51c6252bf8d70958af46022faca39fa8"
EVALUATOR_TEST_SHA256 = "5348b1d61388943e034f7d2ca5865f22630f4b13fd28cfed487dec1738747b66"
LAYERWISE_TEST_SHA256 = "cf8181e623673dfd03c85b317f3f69e58f231a32b66e7db08c9ea40623fb66f1"
SPECS = {
    "t1_t6": "predict_correctness predict_error_persistence",
    "t2": "predict_response",
    "t3": "diagnose_misconception",
    "t5": "predict_prospective_error",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def screen_command() -> str:
    return f'''set -euo pipefail
export PATH="/tmp/studentmodeling-m4-eval-venv/bin:$$PATH"
export PYTHONPATH="$$AMLT_CODE_DIR/scripts$${{PYTHONPATH:+:$$PYTHONPATH}}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/layerwise_formal_screen_fixedu4_v1/test.jsonl
export CAPTURE_ADAPTER={CAPTURE_ADAPTER}
export DECODE_ADAPTER={DECODE_ADAPTER}
mkdir -p "$$HF_HUB_CACHE" "$$(dirname "$$RECURRENT_EVAL")" "$$AMLT_OUTPUT_DIR/eval/$$EVAL_GROUP"
test -f "$$RAW_EVAL"
test -f "$$CAPTURE_ADAPTER/adapter_model.safetensors"
test -f "$$DECODE_ADAPTER/adapter_model.safetensors"
test "$$(sha256sum "$$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = {EVALUATOR_SHA256}
test "$$(sha256sum scripts/layerwise_recurrent_student_model.py | cut -d' ' -f1)" = {LAYERWISE_SHA256}
test "$$(sha256sum tests/test_evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = {EVALUATOR_TEST_SHA256}
test "$$(sha256sum tests/test_layerwise_recurrent_student_model.py | cut -d' ' -f1)" = {LAYERWISE_TEST_SHA256}
test "$$(sha256sum scripts/build_recurrent_evaluation_manifest.py | cut -d' ' -f1)" = d335b0b97471683eb1f317f21af6f0dd36ccca0fe6a8e495838112785ea85a8e
python -m unittest discover -s tests -p 'test_evaluate_recurrent_formal_protocols.py' -v
python -m unittest discover -s tests -p 'test_layerwise_recurrent_student_model.py' -v
python scripts/build_recurrent_evaluation_manifest.py \\
  --input-jsonl "$$RAW_EVAL" \\
  --output-jsonl "$$RECURRENT_EVAL" \\
  --audit-json /tmp/layerwise_formal_screen_fixedu4_v1/audit.json \\
  --max-updates 4 \\
  --seed 20260730
test "$$(sha256sum "$$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
rm -rf "$$LOCAL_MODEL"
mkdir -p "$$LOCAL_MODEL"
python scripts/copy_model_parallel.py \\
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \\
  --destination "$$LOCAL_MODEL" \\
  --workers 16
test -f "$$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u scripts/evaluate_recurrent_formal_protocols.py \\
  --model "$$LOCAL_MODEL" \\
  --adapter "$$CAPTURE_ADAPTER" \\
  --decode-adapter "$$DECODE_ADAPTER" \\
  --state-architecture layerwise \\
  --eval-jsonl "$$RECURRENT_EVAL" \\
  --output-dir "$$AMLT_OUTPUT_DIR/eval/$$EVAL_GROUP" \\
  --state-token-alias '<|vision_pad|>' \\
  --max-samples-per-group 50 \\
  --state-batch-size 2 \\
  --candidate-row-batch-size 2 \\
  --max-length 1536 \\
  --max-new-tokens 128 \\
  --tasks $$TASKS \\
  --seed 42 \\
  --save-unredacted-predictions
cp /tmp/layerwise_formal_screen_fixedu4_v1/audit.json "$$AMLT_OUTPUT_DIR/eval/$$EVAL_GROUP/manifest_audit.json"
printf '%s\n' \\
  'stage=zero_training_layerwise_formal_screen' \\
  'capture_adapter=cf1_visionpad' \\
  'decode_adapter=full_task' \\
  'state_architecture=one_address_layerwise_residual_trajectory' \\
  'raw_history_visible_to_decoder=false' \\
  'persistent_history_kv_cache=false' \\
  'optimizer_steps=0' \\
  > "$$AMLT_OUTPUT_DIR/eval/$$EVAL_GROUP/scope.txt"
'''


def render(base: dict) -> dict:
    source = next(
        (job for job in base.get("jobs") or [] if job.get("name") == "m4-eval-t1-t6-seed42"),
        None,
    )
    if source is None:
        raise ValueError("Formal evaluation template job is missing")
    result = copy.deepcopy(base)
    result["description"] = (
        "Zero-training formal performance screen of CF1 layerwise recurrent state "
        "decoded by the complete-task adapter"
    )
    command = screen_command()
    result["jobs"] = []
    for group, tasks in SPECS.items():
        job = copy.deepcopy(source)
        job["name"] = f"layerwise-formal-screen-{group}-seed42"
        job["command"] = command
        job["sku"] = "80G1-A100"
        job["mpi"] = False
        job["process_count_per_node"] = 1
        submit = job.setdefault("submit_args", {})
        submit["max_attempts"] = 2
        submit["max_run_duration_seconds"] = 172800
        submit["env"] = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "EVAL_GROUP": group,
            "TASKS": tasks,
        }
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
