#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
source_dir="$pipeline/equal_state_path_open_t3/local_round3250_c4"
state_dir="$pipeline/equal_state_path_open_t3/local_round3250_c4_full"
checkpoint="$source_dir/checkpoint-round-3250"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B

mkdir -p "$state_dir/progress"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
heartbeat_pid=""
cleanup() {
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
test -f "$checkpoint/adapter/adapter_model.safetensors" || fail "adapter missing"
test -f "$checkpoint/recurrent_state.pt" || fail "recurrent state missing"
test -f "$checkpoint/checkpoint_completion.json" || fail "checkpoint receipt missing"
test -f "$source_dir/artifact_audit.json" || fail "artifact audit missing"

"$python_bin" - "$checkpoint/checkpoint_completion.json" "$source_dir/artifact_audit.json" <<'PY' \
  || fail "checkpoint and artifact audit disagree"
import json,sys
receipt=json.load(open(sys.argv[1],encoding='utf-8'))
audit=json.load(open(sys.argv[2],encoding='utf-8'))
assert receipt.get('status')=='complete' and receipt.get('round')==3250,receipt
assert receipt.get('world_size')==4,receipt
assert audit.get('status')=='passed' and audit.get('round')==3250,audit
assert audit.get('world_size')==4 and audit.get('sha256')==receipt.get('sha256'),audit
PY

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 60; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat
  sleep 60
done

if [[ ! -s "$state_dir/recurrent_causal_gate.json" ]]; then
  (
    while true; do heartbeat; sleep 60; done
  ) &
  heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise \
      --jsonl data/processed/malrule_c1_clean/test_same_template.jsonl \
      --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
      --output-json "$state_dir/recurrent_causal_gate.json" \
      --state-token-alias '<|vision_pad|>' --max-length 1536 \
      --num-pairs 150 --seed 42 --progress-dir "$state_dir/progress" \
      --progress-key 'equal-open-t3-state-path-lr5e7-round3250-full-n150' \
      >> "$state_dir/live.log" 2>&1 \
    || fail "local full C4 failed"
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
fi

"$python_bin" - "$state_dir/recurrent_causal_gate.json" "$state_dir/summary.json" <<'PY' \
  || fail "full C4 summary invalid"
import json,pathlib,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))['summary']; c=x['clean']
assert x['all_pairs_n']==150 and x['clean_n']>0,x
out={'stage':'layerwise_checkpoint_local_c4','artifact_label':'equal-open-t3-state-path-lr5e7',
     'checkpoint_round':3250,'diagnostic_only':True,'not_a_formal_gate':True,
     'all_pairs_n':x['all_pairs_n'],'clean_n':x['clean_n'],
     'swap':c['swap']['flip_to_A_rate'],'random':c['random']['flip_to_A_rate'],
     'same_rule':c['samemal']['flip_to_A_rate'],'noop':c['noop']['flip_to_A_rate'],
     'D_specific':x['D_specific'],'D_damage':x['D_damage'],
     'scientific_gate_pass':x['scientific_gate_pass'],'checks':x['checks']}
path=pathlib.Path(sys.argv[2]); tmp=path.with_suffix(path.suffix+'.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
print(json.dumps(out,indent=2,sort_keys=True))
PY
touch "$state_dir/complete"
heartbeat
log "equal open-T3 state-path round-3250 full C4 complete"
