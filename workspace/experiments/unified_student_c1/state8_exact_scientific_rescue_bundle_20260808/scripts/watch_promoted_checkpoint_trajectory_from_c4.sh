#!/usr/bin/env bash
set -uo pipefail

: "${CHECKPOINT_ROUND:?CHECKPOINT_ROUND is required}"
: "${SOURCE_C4_SUBDIR:?SOURCE_C4_SUBDIR is required}"
: "${TRAJECTORY_SUBDIR:?TRAJECTORY_SUBDIR is required}"

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
fit_dir="$pipeline/task_retention_promoted_final_fit"
source="$fit_dir/$SOURCE_C4_SUBDIR"
state_dir="$fit_dir/$TRAJECTORY_SUBDIR"
checkpoint="$source/checkpoint-round-$CHECKPOINT_ROUND"
adapter="$checkpoint/adapter"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python

mkdir -p "$state_dir/formal" "$state_dir/t3_semantic"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state_dir/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention" \
  "$state_dir/diagnostic_gate_pass" "$state_dir/diagnostic_gate_failed"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while [[ ! -f "$source/complete" ]]; do
  heartbeat
  [[ ! -f "$source/needs_attention" ]] || fail "source C4 checkpoint audit needs attention"
  sleep 30
done
test -s "$checkpoint/checkpoint_completion.json" || fail "checkpoint receipt missing"
test -s "$adapter/adapter_model.safetensors" || fail "checkpoint adapter missing"
test -s "$checkpoint/recurrent_state.pt" || fail "checkpoint state missing"
test -s "$source/artifact_audit.json" || fail "checkpoint artifact audit missing"
test -s "$source/exact_resume_audit.json" || fail "checkpoint exact-resume audit missing"
test -s "$source/promoted_recipe_audit.json" || fail "checkpoint promoted-recipe audit missing"
test -s "$source/c4/recurrent_causal_gate.json" || fail "checkpoint C4 result missing"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "frozen formal panel drift"

"$python_bin" - "$checkpoint/checkpoint_completion.json" "$source/artifact_audit.json" \
  "$source/exact_resume_audit.json" "$source/promoted_recipe_audit.json" "$CHECKPOINT_ROUND" <<'PY' \
  || fail "checkpoint provenance audit failed"
import json,sys
receipt,artifact,exact,recipe=(json.load(open(path,encoding='utf-8')) for path in sys.argv[1:5])
round_index=int(sys.argv[5])
assert receipt['status']=='complete' and receipt['round']==round_index and receipt['world_size']==16,receipt
assert artifact['status']=='passed' and artifact['round']==round_index and artifact['sha256']==receipt['sha256'],artifact
assert exact['status']=='passed' and exact['checkpoint_round']==round_index and exact['exact_resume_ready'] is True,exact
assert recipe['status']=='passed' and recipe['checkpoint_round']==round_index,recipe
PY

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 30; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat; sleep 30
done
if [[ ! -s "$state_dir/formal/metrics.json" ]]; then
  (while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols.py \
      --model "$model" --adapter "$adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" \
      --output-dir "$state_dir/formal" --state-token-alias '<|vision_pad|>' \
      --max-samples-per-group 25 --max-length 1536 --max-new-tokens 128 \
      --state-batch-size 2 --candidate-row-batch-size 2 \
      --tasks predict_correctness predict_error_persistence diagnose_misconception \
      --seed 42 --save-unredacted-predictions >> "$state_dir/formal/live.log" 2>&1 \
    || fail "round-$CHECKPOINT_ROUND T1/T3/T6 screen failed"
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
fi
flock -u 10

if [[ ! -s "$state_dir/t3_semantic/metrics.json" ]]; then
  PYTHONPATH="$project_dir/scripts:/home/v-yuejiali/fkft/src" \
    "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$state_dir/formal/predictions.jsonl" \
      --eval-jsonl "$eval_jsonl" --output-dir "$state_dir/t3_semantic" \
      --judge-backend trapi --judge-workers 12 --judge-max-retries 12 \
      --judge-timeout-seconds 60 --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
      --judge-audit-samples 75 --seed 42 >> "$state_dir/t3_semantic/live.log" 2>&1 \
    || fail "round-$CHECKPOINT_ROUND T3 semantic judge failed"
fi
"$python_bin" scripts/validate_checkpoint_t3_semantic_result.py \
  --metrics "$state_dir/t3_semantic/metrics.json" \
  --output "$state_dir/t3_semantic_gate.json" --checkpoint-round "$CHECKPOINT_ROUND" \
  >> "$state_dir/controller.log" 2>&1
semantic_rc=$?
(( semantic_rc == 0 || semantic_rc == 2 )) \
  || fail "round-$CHECKPOINT_ROUND semantic validator failed rc=$semantic_rc"

"$python_bin" scripts/summarize_promoted_checkpoint_trajectory.py \
  --metrics "$state_dir/formal/metrics.json" \
  --semantic "$state_dir/t3_semantic_gate.json" \
  --c4 "$source/c4/recurrent_causal_gate.json" \
  --checkpoint-round "$CHECKPOINT_ROUND" \
  --baseline-metrics "$fit_dir/local_round1000_formal_screen/t1_t6_t3/metrics.json" \
  --baseline-semantic "$fit_dir/local_round1000_t3_semantic/t3_semantic_gate.json" \
  --output "$state_dir/summary.json" >> "$state_dir/controller.log" 2>&1
summary_rc=$?
(( summary_rc == 0 || summary_rc == 2 )) \
  || fail "round-$CHECKPOINT_ROUND trajectory summary failed rc=$summary_rc"
if (( summary_rc == 0 )); then
  touch "$state_dir/diagnostic_gate_pass"
else
  touch "$state_dir/diagnostic_gate_failed"
fi
touch "$state_dir/complete"; heartbeat
log "round-$CHECKPOINT_ROUND promoted joint trajectory screen complete summary_rc=$summary_rc"
