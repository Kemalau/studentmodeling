#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
root="$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit/formal_full"
checkpoint="$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit"
python_bin="/home/v-yuejiali/.venvs/studentmodeling/bin/python"
model="/home/v-yuejiali/studentmodeling/models/Qwen3-32B"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
feeds_root="$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint750_formal_full_feeds"

mkdir -p "$root"
cd "$project_dir"
exec 9>"$root/controller.lock"
flock -n 9 || exit 0
echo $$ > "$root/controller.active.pid"
trap 'rm -f "$root/controller.active.pid"' EXIT
rm -f "$root/complete" "$root/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$root/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$root/controller.log"; }
fail(){ log "ERROR: $*"; touch "$root/needs_attention"; exit 1; }

test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B is incomplete"
test -f "$checkpoint/adapter/adapter_model.safetensors" || fail "round-750 adapter is missing"
test -f "$checkpoint/recurrent_state.pt" || fail "round-750 recurrent state is missing"
test -f "$eval_jsonl" || fail "frozen evaluation manifest is missing"

run_group(){
  local group="$1"; shift
  local output="$root/$group"
  if [[ -s "$output/metrics.json" ]]; then
    log "reusing completed full group=$group"
    return 0
  fi
  mkdir -p "$output"
  heartbeat
  (
    while true; do heartbeat; sleep 60; done
  ) &
  local heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" \
      --output-dir "$output" --state-token-alias '<|vision_pad|>' \
      --max-length 1536 --max-new-tokens 128 \
      --state-batch-size 2 --candidate-row-batch-size 2 \
      --tasks "$@" --seed 42 --save-unredacted-predictions \
      >> "$output/live.log" 2>&1
  local rc=$?
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  ((rc==0)) || fail "full group=$group failed rc=$rc"
  test -s "$output/metrics.json" || fail "full group=$group has no metrics"
  log "full group=$group complete"
}

# T1 may already have been started interactively. Wait for that immutable run
# instead of launching a duplicate model on the same GPU. If the process dies
# before writing metrics, retry the same frozen command through this controller.
while tmux has-session -t layerwise_checkpoint750_t1_full 2>/dev/null; do
  heartbeat
  sleep 60
done
run_group t1 predict_correctness
# Prefer the already-submitted three-way Feeds evaluation.  Its controller
# preserves passed siblings and atomically installs full-support T2/T3/T5
# outputs below.  Fall back to this local GPU only if remote recovery is
# explicitly exhausted; never duplicate an active remote job.
while [[ ! -f "$feeds_root/complete" && ! -f "$feeds_root/needs_attention" ]]; do
  heartbeat
  sleep 60
done
run_group t2 predict_response
run_group t5 predict_prospective_error
run_group t3 diagnose_misconception

"$python_bin" scripts/summarize_layerwise_checkpoint750_full_diagnostic.py \
  --formal-root "$root" \
  --t6-metrics "$checkpoint/t6_full/metrics.json" \
  --c4-gate "$checkpoint/c4_full/recurrent_causal_gate.json" \
  --t3-semantic-metrics "$checkpoint/t3_decode_diagnostics/semantic_judge_full75/metrics.json" \
  --output "$root/summary.json" >> "$root/controller.log" 2>&1
summary_rc=$?
# A failed scientific gate is expected diagnostic evidence, not an execution
# failure. The summarizer uses exit 2 while still writing the full receipt.
((summary_rc==0 || summary_rc==2)) || fail "full diagnostic summary failed"
test -s "$root/summary.json" || fail "full diagnostic summary is missing"
touch "$root/complete"
heartbeat
log "round-750 full local diagnostic complete summary_rc=$summary_rc"
