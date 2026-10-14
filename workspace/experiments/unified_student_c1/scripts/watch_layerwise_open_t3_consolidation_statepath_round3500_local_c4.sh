#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/open_t3_consolidation/local_statepath_round3500_c4_full"
checkpoint="$state_dir/checkpoint-round-3500"
adapter="$checkpoint/adapter"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
account=csnewsandfeeds4150361735
container=amulet
prefix=studentmodeling/diagnostics/layerwise_task_retention_service_recovery/equal_t3_t6_open_t3/state_path/lr5e-7_start13000_stop3500_3a6afeecb7949f72f464efcf608402b3a258e88609f4d8d45bf0bbd252a26b16_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a/checkpoint-round-3500

mkdir -p "$state_dir/progress" "$adapter"
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
download_atomic(){
  local name="$1" output="$2" partial="$2.partial.$$"
  rm -f "$partial"
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$name" --file "$partial" \
    --overwrite -o none >> "$state_dir/download.log" 2>&1 || return 1
  mv "$partial" "$output"
}

heartbeat
receipt="$checkpoint/checkpoint_completion.json"
if [[ ! -s "$receipt" ]]; then
  mkdir -p "$checkpoint"
  download_atomic "$prefix/checkpoint_completion.json" "$receipt" \
    || fail "checkpoint receipt download failed"
fi
"$python_bin" - "$receipt" <<'PY' || fail "checkpoint receipt invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x.get('status')=='complete' and x.get('round')==3500,x
assert x.get('world_size')==4,x
assert set(x.get('sha256') or {})=={'adapter_model','recurrent_state'},x
PY

if [[ ! -s "$adapter/adapter_model.safetensors" ]]; then
  download_atomic "$prefix/adapter/adapter_model.safetensors" \
    "$adapter/adapter_model.safetensors" || fail "adapter download failed"
fi
if [[ ! -s "$adapter/adapter_config.json" ]]; then
  download_atomic "$prefix/adapter/adapter_config.json" \
    "$adapter/adapter_config.json" || fail "adapter config download failed"
fi
if [[ ! -s "$checkpoint/recurrent_state.pt" ]]; then
  download_atomic "$prefix/recurrent_state.pt" "$checkpoint/recurrent_state.pt" \
    || fail "recurrent-state download failed"
fi

"$python_bin" - "$receipt" "$adapter/adapter_model.safetensors" \
  "$checkpoint/recurrent_state.pt" "$state_dir/artifact_audit.json" <<'PY' \
  || fail "checkpoint hash audit failed"
import hashlib,json,pathlib,sys
receipt=json.load(open(sys.argv[1],encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(sys.argv[2]),'recurrent_state':digest(sys.argv[3])}
assert observed==receipt['sha256'],(observed,receipt)
out={'stage':'layerwise_checkpoint_local_artifact_audit','artifact_label':'open-t3-consolidation-statepath-lr5e7',
     'status':'passed','round':3500,'world_size':4,'sha256':observed}
path=pathlib.Path(sys.argv[4]); temporary=path.with_suffix(path.suffix+'.partial')
temporary.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); temporary.replace(path)
PY

test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B missing"
test -f data/processed/malrule_c1_clean/test_same_template.jsonl || fail "C4 source missing"
test -f manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json || fail "C4 manifest missing"

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 60; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat
  sleep 60
done

if [[ ! -s "$state_dir/recurrent_causal_gate.json" ]]; then
  ( while true; do heartbeat; sleep 60; done ) &
  heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
      --model "$model" --adapter "$adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise \
      --jsonl data/processed/malrule_c1_clean/test_same_template.jsonl \
      --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
      --output-json "$state_dir/recurrent_causal_gate.json" \
      --state-token-alias '<|vision_pad|>' --max-length 1536 \
      --num-pairs 150 --seed 42 --progress-dir "$state_dir/progress" \
      --progress-key 'open-t3-consolidation-statepath-lr5e7-round3500-full-n150' \
      >> "$state_dir/live.log" 2>&1 || fail "local full C4 failed"
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
fi

"$python_bin" - "$state_dir/recurrent_causal_gate.json" "$state_dir/summary.json" <<'PY' \
  || fail "full C4 summary invalid"
import json,pathlib,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))['summary']; c=x['clean']
assert x['all_pairs_n']==150 and x['clean_n']>0,x
out={'stage':'layerwise_checkpoint_local_c4','artifact_label':'open-t3-consolidation-statepath-lr5e7',
     'checkpoint_round':3500,'diagnostic_only':True,'not_a_formal_gate':True,
     'all_pairs_n':x['all_pairs_n'],'clean_n':x['clean_n'],
     'swap':c['swap']['flip_to_A_rate'],'random':c['random']['flip_to_A_rate'],
     'same_rule':c['samemal']['flip_to_A_rate'],'noop':c['noop']['flip_to_A_rate'],
     'D_specific':x['D_specific'],'D_damage':x['D_damage'],
     'scientific_gate_pass':x['scientific_gate_pass'],'checks':x['checks']}
path=pathlib.Path(sys.argv[2]); temporary=path.with_suffix(path.suffix+'.partial')
temporary.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); temporary.replace(path)
print(json.dumps(out,indent=2,sort_keys=True))
PY
touch "$state_dir/complete"
heartbeat
log "open-T3 consolidation state-path round-3500 full C4 complete"
