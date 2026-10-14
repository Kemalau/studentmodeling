#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
base="results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan"
artifact="$base/calibration_artifacts/layerwise-eedi-full-open-t3-lr-scan-recovery1-seed42-20260802/layerwise-eedi-full-open-t3-lr2e6-seed42/model"
output="$base/user_candidate_full_eval_20260803"
python_bin="/home/v-yuejiali/.venvs/studentmodeling/bin/python"

cd "$project_dir"
mkdir -p "$output/t2_t3_t5_t6"

# Serialize with the already-running T1 process on the single local A100.
exec 9>results/local_gpu_readonly_preflight.lock
flock 9

test -s "$output/t1/metrics.json"
test -s "$artifact/adapter_model.safetensors"
test -s "$artifact/recurrent_state.pt"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$python_bin" -u scripts/evaluate_recurrent_formal_protocols.py \
  --model /home/v-yuejiali/studentmodeling/models/Qwen3-32B \
  --adapter "$artifact" \
  --recurrent-state "$artifact/recurrent_state.pt" \
  --state-architecture layerwise \
  --eval-jsonl data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl \
  --output-dir "$output/t2_t3_t5_t6" \
  --state-token-alias '<|vision_pad|>' \
  --max-length 1536 \
  --max-new-tokens 128 \
  --state-batch-size 2 \
  --candidate-row-batch-size 2 \
  --tasks predict_response diagnose_misconception predict_prospective_error predict_error_persistence \
  --seed 42 \
  --save-unredacted-predictions \
  2>&1 | tee -a "$output/t2_t3_t5_t6/live.log"

jq -s \
  '.[0] * {metrics_by_group: (.[0].metrics_by_group + .[1].metrics_by_group), sampled_rows: (.[0].sampled_rows + .[1].sampled_rows)}' \
  "$output/t1/metrics.json" "$output/t2_t3_t5_t6/metrics.json" \
  > "$output/metrics.all.tmp.json"
mv "$output/metrics.all.tmp.json" "$output/metrics.json"

"$python_bin" scripts/summarize_method_small_formal_metrics.py \
  "$output/metrics.json" --output "$output/headline.json"

"$python_bin" - "$output/headline.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"T1": 10913, "T2": 3693, "T3": 1494, "T5": 4600, "T6": 976}
assert {key: int(value["n"]) for key, value in payload.items()} == expected, payload
PY

touch "$output/complete"
