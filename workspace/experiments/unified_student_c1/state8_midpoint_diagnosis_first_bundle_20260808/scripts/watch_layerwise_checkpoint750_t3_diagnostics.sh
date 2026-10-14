#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
checkpoint="$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit"
state_dir="$checkpoint/t3_decode_diagnostics"
formal_dir="$checkpoint/formal_screen"
python_bin="/home/v-yuejiali/.venvs/studentmodeling/bin/python"
model="/home/v-yuejiali/studentmodeling/models/Qwen3-32B"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
specialist="$project_dir/results/qwen3_32b_eedi_t3_qlora_1epoch_len1536/final_adapter"

mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while [[ ! -f "$formal_dir/complete" ]]; do
  heartbeat
  [[ ! -f "$formal_dir/needs_attention" ]] || fail "checkpoint750 formal screen needs attention"
  sleep 30
done
test -f "$specialist/adapter_model.safetensors" || fail "local T3 specialist missing"

run_condition(){
  local name="$1" decode_adapter="$2" output="$state_dir/$1"
  [[ -s "$output/metrics.json" ]] && { log "reusing T3 diagnostic=$name"; return 0; }
  mkdir -p "$output"
  local decode_args=()
  [[ -n "$decode_adapter" ]] && decode_args=(--decode-adapter "$decode_adapter")
  heartbeat
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_t3_constrained.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      "${decode_args[@]}" --eval-jsonl "$eval_jsonl" --output-dir "$output" \
      --state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \
      --max-length 1536 --max-content-tokens 112 --state-batch-size 2 --seed 42 \
      >> "$output/live.log" 2>&1 || fail "T3 diagnostic=$name failed"
  test -s "$output/metrics.json" || fail "T3 diagnostic=$name has no metrics"
  log "T3 diagnostic=$name complete"
}

run_condition same_adapter_constrained ""
run_condition eedi_specialist_decoder_constrained "$specialist"
touch "$state_dir/complete"
heartbeat
log "checkpoint750 T3 constrained decode diagnostics complete"
