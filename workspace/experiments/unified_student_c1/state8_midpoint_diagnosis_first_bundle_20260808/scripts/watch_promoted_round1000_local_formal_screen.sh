#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
checkpoint_root="$pipeline/task_retention_promoted_final_fit/local_round1000_c4"
checkpoint="$checkpoint_root/checkpoint-round-1000"
state_dir="$pipeline/task_retention_promoted_final_fit/local_round1000_formal_screen"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state_dir"
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
rm -f "$state_dir/needs_attention" "$state_dir/complete"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

heartbeat
test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B is incomplete"
test -f "$eval_jsonl" || fail "frozen recurrent evaluation manifest is missing"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "frozen recurrent evaluation manifest drift"
test -f "$checkpoint/adapter/adapter_model.safetensors" || fail "round-1000 adapter missing"
test -f "$checkpoint/recurrent_state.pt" || fail "round-1000 recurrent state missing"
test -f "$checkpoint/checkpoint_completion.json" || fail "round-1000 receipt missing"
test -f "$checkpoint_root/exact_resume_audit.json" || fail "exact-resume audit missing"
test -f "$checkpoint_root/promoted_recipe_audit.json" || fail "promoted recipe audit missing"

"$python_bin" - "$checkpoint" "$checkpoint_root/exact_resume_audit.json" \
  "$checkpoint_root/promoted_recipe_audit.json" <<'PY' \
  || fail "round-1000 checkpoint or promotion audit invalid"
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); receipt=json.load(open(root/'checkpoint_completion.json'))
exact=json.load(open(sys.argv[2])); recipe=json.load(open(sys.argv[3]))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(root/'adapter/adapter_model.safetensors'),
          'recurrent_state':digest(root/'recurrent_state.pt')}
assert receipt=={'status':'complete','round':1000,'world_size':16,'sha256':observed},receipt
assert exact['status']=='passed' and exact['exact_resume_ready'] is True,exact
assert exact['checkpoint_round']==1000 and exact['continuation_source_round']==750,exact
assert recipe['status']=='passed' and recipe['candidate_id']=='eedi_full_open_t3__state_path__lr2e-06',recipe
PY

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 30; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat
  sleep 30
done

run_group(){
  local group="$1"; shift
  local output="$state_dir/$group" rc
  if [[ -s "$output/metrics.json" ]]; then
    log "reusing completed promoted round-1000 formal group=$group"
    return 0
  fi
  mkdir -p "$output"
  heartbeat
  (
    while true; do heartbeat; sleep 60; done
  ) &
  heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" \
      --output-dir "$output" --state-token-alias '<|vision_pad|>' \
      --max-samples-per-group 25 --max-length 1536 --max-new-tokens 128 \
      --state-batch-size 2 --candidate-row-batch-size 2 \
      --tasks "$@" --seed 42 --save-unredacted-predictions \
      >> "$output/live.log" 2>&1
  rc=$?
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
  ((rc==0)) || fail "promoted round-1000 formal group=$group failed rc=$rc"
  test -s "$output/metrics.json" || fail "promoted round-1000 formal group=$group has no metrics"
  log "promoted round-1000 formal group=$group complete"
}

run_group t1_t6_t3 predict_correctness predict_error_persistence diagnose_misconception
run_group t2_t5 predict_response predict_prospective_error

"$python_bin" scripts/summarize_layerwise_checkpoint250_formal_screen.py \
  --root "$state_dir" --output "$state_dir/summary.json" --checkpoint-round 1000 \
  >> "$state_dir/controller.log" 2>&1
summary_rc=$?
((summary_rc==0 || summary_rc==2)) || fail "promoted round-1000 formal summary invalid"
test -s "$state_dir/summary.json" || fail "promoted round-1000 formal summary missing"
if ((summary_rc==0)); then touch "$state_dir/diagnostic_gate_pass"; else touch "$state_dir/diagnostic_gate_failed"; fi
touch "$state_dir/complete"
heartbeat
log "promoted round-1000 local formal screen complete summary_rc=$summary_rc"
