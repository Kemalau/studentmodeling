#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/task_retention_promoted_final_fit/local_round1250_trajectory"
checkpoint="$state_dir/checkpoint-round-1250"
adapter="$checkpoint/adapter"
remote_prefix=studentmodeling/checkpoints/layerwise_unified_world_model_seed42_taskretention_eedi_full_open_t3_state_path_v1_20260802/checkpoint-round-1250
promotion="$pipeline/task_retention_promoted_final_fit/selected.json"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
c4_jsonl="$project_dir/data/processed/malrule_c1_clean/test_same_template.jsonl"
parent_pairs="$project_dir/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"
pair_manifest="$state_dir/c4_pairs_n50.json"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
account=csnewsandfeeds4150361735
container=amulet
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds

mkdir -p "$adapter" "$state_dir/formal" "$state_dir/t3_semantic" "$state_dir/c4/progress"
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
download_atomic(){
  local name="$1" output="$2" partial="$2.partial.$$"
  rm -f "$partial"
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$name" --file "$partial" \
    --overwrite -o none >> "$state_dir/download.log" 2>&1 || return 1
  mv "$partial" "$output"
}

heartbeat
test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B missing"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "frozen formal panel drift"
test -s "$promotion" || fail "promoted manifest missing"
test -s "$c4_jsonl" || fail "C4 data missing"
test -s "$parent_pairs" || fail "C4 pair manifest missing"

receipt="$checkpoint/checkpoint_completion.json"
for target in checkpoint_completion.json adapter/adapter_config.json adapter/adapter_model.safetensors recurrent_state.pt; do
  output="$checkpoint/$target"
  mkdir -p "$(dirname "$output")"
  [[ -s "$output" ]] || download_atomic "$remote_prefix/$target" "$output" \
    || fail "download failed target=$target"
done

"$python_bin" - "$checkpoint" "$promotion" <<'PY' || fail "round-1250 artifact audit failed"
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); promotion=pathlib.Path(sys.argv[2])
receipt=json.load(open(root/'checkpoint_completion.json'))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(root/'adapter/adapter_model.safetensors'),
          'recurrent_state':digest(root/'recurrent_state.pt')}
assert receipt=={'status':'complete','round':1250,'world_size':16,'sha256':observed},receipt
assert hashlib.sha256(promotion.read_bytes()).hexdigest()=='adc6eacca29cf22c554875d0f1e5cd8e6cbd6722966228b3aafe8a3e62ca0ee7'
PY
"$python_bin" scripts/audit_layerwise_distributed_checkpoint.py \
  --checkpoint "$checkpoint/recurrent_state.pt" --expected-round 1250 \
  --output "$state_dir/exact_resume_audit.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "round-1250 distributed checkpoint audit failed"
"$python_bin" scripts/audit_layerwise_promoted_checkpoint.py \
  --checkpoint "$checkpoint/recurrent_state.pt" --manifest "$promotion" \
  --expected-round 1250 --output "$state_dir/promoted_recipe_audit.json" \
  >> "$state_dir/controller.log" 2>&1 || fail "round-1250 promoted recipe audit failed"

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
    || fail "round-1250 T1/T3/T6 screen failed"
  kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; heartbeat_pid=""
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
    || fail "round-1250 T3 semantic judge failed"
fi
"$judge_python" scripts/validate_checkpoint_t3_semantic_result.py \
  --metrics "$state_dir/t3_semantic/metrics.json" \
  --output "$state_dir/t3_semantic_gate.json" --checkpoint-round 1250 \
  >> "$state_dir/controller.log" 2>&1
semantic_rc=$?
((semantic_rc==0 || semantic_rc==2)) || fail "round-1250 semantic validator failed rc=$semantic_rc"

if [[ ! -s "$pair_manifest" ]]; then
  "$python_bin" - "$parent_pairs" "$pair_manifest" <<'PY' || fail "C4 pair derivation failed"
import hashlib,json,pathlib,sys
source=pathlib.Path(sys.argv[1]); x=json.loads(source.read_text())
assert x['num_pairs']==150 and len(x['pairs'])==150
out={**x,'num_pairs':50,'pairs':x['pairs'][:50],'diagnostic_only':True,
     'selection':'ordered_prefix','parent_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
path=pathlib.Path(sys.argv[2]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
fi

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 30; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat; sleep 30
done
if [[ ! -s "$state_dir/c4/recurrent_causal_gate.json" ]]; then
  (while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
      --model "$model" --adapter "$adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --jsonl "$c4_jsonl" \
      --pair-manifest "$pair_manifest" \
      --output-json "$state_dir/c4/recurrent_causal_gate.json" \
      --state-token-alias '<|vision_pad|>' --max-length 1536 \
      --num-pairs 50 --seed 42 --progress-dir "$state_dir/c4/progress" \
      --progress-key promoted-round1250-local-n50 \
      >> "$state_dir/c4/live.log" 2>&1 || fail "round-1250 C4 n50 failed"
  kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; heartbeat_pid=""
fi
flock -u 10

"$python_bin" scripts/summarize_promoted_checkpoint_trajectory.py \
  --metrics "$state_dir/formal/metrics.json" \
  --semantic "$state_dir/t3_semantic_gate.json" \
  --c4 "$state_dir/c4/recurrent_causal_gate.json" --checkpoint-round 1250 \
  --baseline-metrics "$pipeline/task_retention_promoted_final_fit/local_round1000_formal_screen/t1_t6_t3/metrics.json" \
  --baseline-semantic "$pipeline/task_retention_promoted_final_fit/local_round1000_t3_semantic/t3_semantic_gate.json" \
  --output "$state_dir/summary.json" >> "$state_dir/controller.log" 2>&1
summary_rc=$?
((summary_rc==0 || summary_rc==2)) || fail "round-1250 trajectory summary failed rc=$summary_rc"
if ((summary_rc==0)); then touch "$state_dir/diagnostic_gate_pass"; else touch "$state_dir/diagnostic_gate_failed"; fi
touch "$state_dir/complete"; heartbeat
log "round-1250 promoted trajectory screen complete summary_rc=$summary_rc"
