#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/v-yuejiali/.venvs/studentmodeling/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/v-yuejiali/studentmodeling/models/Qwen3-0.6B}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/processed/malrule_c1_tiny}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/qwen3_0p6b_malrule_next_response_lora_s20}"

cd "$PROJECT_ROOT"

if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
  "$PYTHON_BIN" scripts/prepare_malrule_c1.py \
    --repo data/raw/malrulelib \
    --out-dir "$DATA_DIR" \
    --train-raw 240 \
    --val-raw 48 \
    --test-same-raw 60 \
    --test-cross-raw 60 \
    --probe-seeds 40 \
    --seed 42
fi

"$PYTHON_BIN" scripts/train_c1_lora.py \
  --model "$MODEL_PATH" \
  --train-jsonl "$DATA_DIR/train.jsonl" \
  --valid-jsonl "$DATA_DIR/val.jsonl" \
  --task-filter malrule_next_response \
  --output-dir "$OUTPUT_DIR" \
  --max-length 512 \
  --max-train-samples 240 \
  --max-valid-samples 48 \
  --max-steps 20 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 2e-4 \
  --load-4bit \
  --save-final-only

"$PYTHON_BIN" scripts/evaluate_c1.py \
  --train-jsonl "$DATA_DIR/train.jsonl" \
  --eval-jsonl "$DATA_DIR/test_same_template.jsonl" \
  --output-json "${OUTPUT_DIR}_eval_same_n30.json" \
  --model "$MODEL_PATH" \
  --adapter "$OUTPUT_DIR/final_adapter" \
  --tasks malrule_next_response \
  --max-samples-per-task 30 \
  --max-length 512 \
  --batch-size 16 \
  --progress-every 5
