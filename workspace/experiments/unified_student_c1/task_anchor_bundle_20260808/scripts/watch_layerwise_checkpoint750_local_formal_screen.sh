#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
checkpoint="$pipeline/checkpoint750_local_audit"
state_dir="$checkpoint/formal_screen"
python_bin="/home/v-yuejiali/.venvs/studentmodeling/bin/python"
model="/home/v-yuejiali/studentmodeling/models/Qwen3-32B"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

heartbeat
test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B is incomplete"
test -f "$eval_jsonl" || fail "frozen recurrent evaluation manifest is missing"
test -f "$checkpoint/adapter/adapter_model.safetensors" || fail "round-750 adapter is missing"
test -f "$checkpoint/recurrent_state.pt" || fail "round-750 recurrent state is missing"
test -f "$checkpoint/checkpoint_completion.json" || fail "round-750 completion receipt is missing"

"$python_bin" - "$checkpoint" <<'PY' || fail "round-750 checkpoint receipt invalid"
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
receipt=json.load(open(root/'checkpoint_completion.json',encoding='utf-8'))
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()
assert receipt.get('status')=='complete' and receipt.get('round')==750,receipt
assert receipt.get('world_size')==16,receipt
assert receipt['sha256']['adapter_model']==digest(root/'adapter/adapter_model.safetensors')
assert receipt['sha256']['recurrent_state']==digest(root/'recurrent_state.pt')
PY

run_group(){
  local group="$1"; shift
  local output="$state_dir/$group"
  if [[ -s "$output/metrics.json" ]]; then
    log "reusing completed round-750 formal group=$group"
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
      --max-samples-per-group 25 --max-length 1536 --max-new-tokens 128 \
      --state-batch-size 2 --candidate-row-batch-size 2 \
      --tasks "$@" --seed 42 --save-unredacted-predictions \
      >> "$output/live.log" 2>&1
  local rc=$?
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  ((rc==0)) || fail "round-750 formal group=$group failed rc=$rc"
  test -s "$output/metrics.json" || fail "round-750 formal group=$group has no metrics"
  log "round-750 formal group=$group complete"
}

run_group t1_t6_t3 predict_correctness predict_error_persistence diagnose_misconception
run_group t2_t5 predict_response predict_prospective_error

"$python_bin" scripts/summarize_layerwise_checkpoint250_formal_screen.py \
  --root "$state_dir" --output "$state_dir/summary.json" --checkpoint-round 750 \
  >> "$state_dir/controller.log" 2>&1
summary_rc=$?
# A negative diagnostic gate is a scientific result, not an infrastructure
# failure.  The summarizer uses exit 2 for that case but must still write a
# complete, auditable summary.
((summary_rc==0 || summary_rc==2)) || fail "round-750 formal summary invalid"
test -s "$state_dir/summary.json" || fail "round-750 formal summary missing"
touch "$state_dir/complete"
heartbeat
log "round-750 local formal screen complete summary_rc=$summary_rc"
