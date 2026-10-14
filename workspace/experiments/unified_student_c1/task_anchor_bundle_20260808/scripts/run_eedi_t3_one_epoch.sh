#!/usr/bin/env bash
set -euo pipefail

PY=/home/v-yuejiali/.venvs/studentmodeling/bin/python3
ROOT=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
MODEL=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
DATA="$ROOT/data/processed/eedi_t3_one_epoch"
OUT="$ROOT/results/qwen3_32b_eedi_t3_qlora_1epoch_len1536"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[runner] training start $(date -u +%FT%TZ)"
/usr/bin/time -f "[runner] training elapsed=%e" "$PY" "$ROOT/scripts/train_c1_lora.py" \
    --model "$MODEL" \
    --train-jsonl "$DATA/train.jsonl" \
    --output-dir "$OUT" \
    --max-length 1536 \
    --max-train-samples 10000 \
    --max-steps 2500 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --load-4bit \
    --state-token-alias '<|vision_pad|>' \
    --supervise-eos \
    --logging-steps 10 \
    --save-steps 500
echo "[runner] training done $(date -u +%FT%TZ)"

echo "[runner] closed-set evaluation start $(date -u +%FT%TZ)"
/usr/bin/time -f "[runner] evaluation elapsed=%e" "$PY" "$ROOT/scripts/evaluate_c1_cached.py" \
    --eval-jsonl "$DATA/test.jsonl" \
    --output-json "$OUT/eedi_closed_set_results.json" \
    --model "$MODEL" \
    --adapter "$OUT/final_adapter" \
    --tasks diagnose_misconception \
    --max-length 1536 \
    --state-token-alias '<|vision_pad|>' \
    --prefill-batch-size 6 \
    --label-only \
    --progress-every 100
echo "[runner] complete $(date -u +%FT%TZ)"
