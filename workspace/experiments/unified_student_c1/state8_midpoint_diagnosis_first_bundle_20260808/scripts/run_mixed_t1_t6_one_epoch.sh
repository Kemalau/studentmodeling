#!/usr/bin/env bash
set -euo pipefail

PY=/home/v-yuejiali/.venvs/studentmodeling/bin/python3
ROOT=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
MODEL=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
DATA="$ROOT/data/processed/mixed_t1_t6_balanced"
OUT="$ROOT/results/qwen3_32b_mixed_t1_t6_balanced_open_qlora_1epoch_len1536"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

resume_args=()
latest_checkpoint=$(find "$OUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)
if [[ -n "$latest_checkpoint" ]]; then
    resume_args=(--resume-from-checkpoint "$OUT/$latest_checkpoint")
    echo "[runner] resuming from $latest_checkpoint"
fi

echo "[runner] training start $(date -u +%FT%TZ)"
/usr/bin/time -f "[runner] training elapsed=%e" "$PY" "$ROOT/scripts/train_c1_lora.py" \
    --model "$MODEL" \
    --train-jsonl "$DATA/train.jsonl" \
    --output-dir "$OUT" \
    --max-length 1536 \
    --max-train-samples 120000 \
    --max-steps 30000 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --load-4bit \
    --state-token-alias '<|vision_pad|>' \
    --supervise-eos \
    --logging-steps 20 \
    --save-steps 2500 \
    "${resume_args[@]}"
echo "[runner] training done $(date -u +%FT%TZ)"

echo "[runner] full evaluation start $(date -u +%FT%TZ)"
/usr/bin/time -f "[runner] evaluation elapsed=%e" "$PY" "$ROOT/scripts/evaluate_c1_cached.py" \
    --eval-jsonl "$DATA/test.jsonl" \
    --output-json "$OUT/full_fast_eval.json" \
    --model "$MODEL" \
    --adapter "$OUT/final_adapter" \
    --max-length 1536 \
    --state-token-alias '<|vision_pad|>' \
    --prefill-batch-size 20 \
    --large-label-prefill-batch-size 6 \
    --label-only \
    --progress-every 100
echo "[runner] complete $(date -u +%FT%TZ)"
