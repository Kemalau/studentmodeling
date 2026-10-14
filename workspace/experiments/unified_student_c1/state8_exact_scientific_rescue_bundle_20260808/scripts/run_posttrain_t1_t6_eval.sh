#!/usr/bin/env bash
set -euo pipefail

BASE=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
PYTHON=/home/v-yuejiali/.venvs/studentmodeling/bin/python3
MODEL=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
RUN_DIR="$BASE/results/qwen3_32b_mixed_t1_t6_balanced_open_qlora_s1200_len1536"
ADAPTER="$RUN_DIR/final_adapter"
EVAL_DIR="$RUN_DIR/evaluation"
BASELINE_DIR="$BASE/results/simple_t1_t6_baselines"
SMOKE_DIR="$BASELINE_DIR/eval300"

mkdir -p "$EVAL_DIR"
while [[ ! -f "$ADAPTER/adapter_config.json" ]]; do
  sleep 60
done

export CUDA_VISIBLE_DEVICES=0
"$PYTHON" "$BASE/scripts/make_task_capped_jsonl.py" \
  --input "$BASE/data/processed/mixed_t1_t6_balanced/test.jsonl" \
  --output "$EVAL_DIR/test_300_per_task.jsonl" \
  --per-task 300 \
  > "$EVAL_DIR/make_eval300.log" 2>&1

"$PYTHON" "$BASE/scripts/run_simple_t1_t6_baselines.py" \
  --train-jsonl "$BASE/data/processed/mixed_t1_t6_balanced/train.jsonl" \
  --test-jsonl "$EVAL_DIR/test_300_per_task.jsonl" \
  --out-dir "$SMOKE_DIR" \
  > "$EVAL_DIR/simple_baselines_300.log" 2>&1

"$PYTHON" -u "$BASE/scripts/evaluate_c1.py" \
  --train-jsonl "$BASE/data/processed/mixed_t1_t6_balanced/train.jsonl" \
  --eval-jsonl "$EVAL_DIR/test_300_per_task.jsonl" \
  --output-json "$EVAL_DIR/qwen_eval_300_per_task.json" \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --max-length 1536 \
  --batch-size 4 \
  --progress-every 50 \
  --state-token-alias '<|vision_pad|>' \
  --load-4bit \
  > "$EVAL_DIR/qwen_eval_300_per_task.log" 2>&1

"$PYTHON" "$BASE/scripts/merge_qwen_evaluation_matrix.py" \
  --baseline-json "$SMOKE_DIR/simple_baseline_results.json" \
  --qwen-json "$EVAL_DIR/qwen_eval_300_per_task.json" \
  --out-dir "$SMOKE_DIR" \
  > "$EVAL_DIR/merge_matrix_300.log" 2>&1

"$PYTHON" -u "$BASE/scripts/evaluate_c1.py" \
  --train-jsonl "$BASE/data/processed/mixed_t1_t6_balanced/train.jsonl" \
  --eval-jsonl "$BASE/data/processed/mixed_t1_t6_balanced/test.jsonl" \
  --output-json "$EVAL_DIR/qwen_eval_full.json" \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --max-length 1536 \
  --batch-size 4 \
  --progress-every 100 \
  --state-token-alias '<|vision_pad|>' \
  --load-4bit \
  > "$EVAL_DIR/qwen_eval_full.log" 2>&1

"$PYTHON" "$BASE/scripts/merge_qwen_evaluation_matrix.py" \
  --baseline-json "$BASELINE_DIR/simple_baseline_results.json" \
  --qwen-json "$EVAL_DIR/qwen_eval_full.json" \
  --out-dir "$BASELINE_DIR" \
  > "$EVAL_DIR/merge_matrix_full.log" 2>&1
