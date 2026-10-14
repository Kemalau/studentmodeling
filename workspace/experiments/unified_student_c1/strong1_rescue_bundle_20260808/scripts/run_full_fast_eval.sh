#!/usr/bin/env bash
set -euo pipefail

PY=/home/v-yuejiali/.venvs/studentmodeling/bin/python3
ROOT=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
OUT="$ROOT/results/qwen3_32b_mixed_t1_t6_balanced_open_qlora_s1200_len1536/full_fast_eval"
MODEL=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
ADAPTER="$ROOT/results/qwen3_32b_mixed_t1_t6_balanced_open_qlora_s1200_len1536/final_adapter"
DATA="$ROOT/data/processed/mixed_t1_t6_balanced/test.jsonl"
EVALUATOR="$ROOT/scripts/evaluate_c1_cached.py"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/tasks"

run_task() {
    local task=$1
    local batch_size=$2
    echo "[runner] start task=$task batch=$batch_size $(date -u +%FT%TZ)"
    /usr/bin/time -f "[runner] task=$task elapsed=%e" "$PY" "$EVALUATOR" \
        --eval-jsonl "$DATA" \
        --output-json "$OUT/tasks/$task.json" \
        --model "$MODEL" \
        --adapter "$ADAPTER" \
        --tasks "$task" \
        --max-length 1536 \
        --state-token-alias '<|vision_pad|>' \
        --prefill-batch-size "$batch_size" \
        --label-only \
        --progress-every 100 \
        >"$OUT/tasks/$task.log" 2>&1
    echo "[runner] done task=$task $(date -u +%FT%TZ)"
}

run_task diagnose_misconception 6
run_task estimate_mastery 20
run_task predict_correctness 20
run_task predict_error_persistence 20
run_task predict_prospective_error 20
run_task predict_response 20

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
task_files = sorted((output_dir / "tasks").glob("*.json"))
combined = json.loads(task_files[0].read_text())
combined["model"]["metrics"] = {}
for task_file in task_files:
    result = json.loads(task_file.read_text())
    combined["model"]["metrics"].update(result["model"]["metrics"])
(output_dir / "qwen_full_label_scores.json").write_text(
    json.dumps(combined, indent=2, ensure_ascii=False) + "\n"
)
PY

echo "[runner] complete $(date -u +%FT%TZ)"
