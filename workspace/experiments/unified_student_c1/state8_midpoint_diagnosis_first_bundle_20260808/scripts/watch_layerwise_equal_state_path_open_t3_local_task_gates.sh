#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
owner="$pipeline/equal_state_path_open_t3"
source_dir="$owner/local_round3250_c4"
checkpoint="$source_dir/checkpoint-round-3250"
state_dir="$owner/local_round3250_task_gates"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state_dir/formal_screen" "$state_dir/t6_full"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
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
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

heartbeat
test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B missing"
test -f "$eval_jsonl" || fail "frozen formal evaluation manifest missing"
test -f "$checkpoint/adapter/adapter_model.safetensors" || fail "adapter missing"
test -f "$checkpoint/recurrent_state.pt" || fail "recurrent state missing"
test -f "$checkpoint/checkpoint_completion.json" || fail "checkpoint receipt missing"
test -f "$source_dir/artifact_audit.json" || fail "artifact audit missing"
test -f "$owner/local_round3250_c4_full/summary.json" || fail "full C4 summary missing"

"$python_bin" - "$checkpoint/checkpoint_completion.json" "$source_dir/artifact_audit.json" \
  "$owner/local_round3250_c4_full/summary.json" <<'PY' || fail "candidate artifact/C4 audit failed"
import json,sys
receipt=json.load(open(sys.argv[1],encoding='utf-8'))
audit=json.load(open(sys.argv[2],encoding='utf-8'))
c4=json.load(open(sys.argv[3],encoding='utf-8'))
assert receipt.get('status')=='complete' and receipt.get('round')==3250,receipt
assert receipt.get('world_size')==4 and audit.get('sha256')==receipt.get('sha256'),audit
assert audit.get('status')=='passed' and c4.get('all_pairs_n')==150,(audit,c4)
assert c4.get('scientific_gate_pass') is True,c4
PY

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 60; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat
  sleep 60
done

run_eval(){
  local output="$1"; shift
  if [[ -s "$output/metrics.json" ]]; then return 0; fi
  (
    while true; do heartbeat; sleep 60; done
  ) &
  heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" \
      --output-dir "$output" --state-token-alias '<|vision_pad|>' \
      --max-length 1536 --max-new-tokens 128 --state-batch-size 2 \
      --candidate-row-batch-size 2 --seed 42 --save-unredacted-predictions "$@" \
      >> "$output/live.log" 2>&1 || return 1
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
  test -s "$output/metrics.json"
}

run_eval "$state_dir/formal_screen" --max-samples-per-group 25 \
  || fail "matched formal screen failed"
log "formal screen complete"
run_eval "$state_dir/t6_full" --tasks predict_error_persistence \
  || fail "full T6 evaluation failed"
log "full T6 evaluation complete"
touch "$state_dir/complete"
heartbeat
