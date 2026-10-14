#!/usr/bin/env bash
set -euo pipefail

# Diagnostic-only C4 probe at the first 25%-state mixed-stage checkpoint.
# It never mutates training state and is intentionally excluded from the
# frozen promotion curve (250, 1000, 2000, 3000, 4000, 5000).
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
step=1250
state="$project/results/state8_strong1_widen8_bridge_20260808/single_step1250_c4"
checkpoint="$state/checkpoint-$step"
remote_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4
remote="$remote_root/checkpoint-$step"
account=csnewsandfeeds4150361735
container=amulet
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
c4_jsonl="$project/data/processed/malrule_c1_clean/test_same_template.jsonl"
parent_manifest="$project/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"

mkdir -p "$state/c4/progress" "$checkpoint/adapter"
cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

download_atomic(){
  local blob="$1" output="$2" partial="$2.partial.$$"
  mkdir -p "$(dirname "$output")"; rm -f "$partial"
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$blob" --file "$partial" \
    --overwrite -o none >> "$state/download.log" 2>&1 || return 1
  mv "$partial" "$output"
}

test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B missing"
test -f "$c4_jsonl" || fail "frozen C4 data missing"
test -f "$parent_manifest" || fail "frozen C4 manifest missing"
heartbeat

receipt="$checkpoint/checkpoint_completion.json"
while [[ ! -s "$receipt" ]]; do
  heartbeat
  exists="$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$remote/checkpoint_completion.json" \
    --query exists -o tsv 2>/dev/null || true)"
  if [[ "$exists" == true ]]; then
    download_atomic "$remote/checkpoint_completion.json" "$receipt" \
      || { sleep 30; continue; }
  else
    sleep 30
  fi
done

for file in adapter/adapter_model.safetensors adapter/adapter_config.json recurrent_state.pt; do
  [[ -s "$checkpoint/$file" ]] || download_atomic "$remote/$file" "$checkpoint/$file" \
    || fail "checkpoint download failed file=$file"
done

"$python_bin" - "$receipt" "$checkpoint/adapter/adapter_model.safetensors" \
  "$checkpoint/recurrent_state.pt" "$state/artifact_audit.json" "$step" <<'PY' \
  || fail "checkpoint audit failed"
import collections,hashlib,json,pathlib,sys,torch
receipt=json.load(open(sys.argv[1])); step=int(sys.argv[5])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(sys.argv[2]),'recurrent_state':digest(sys.argv[3])}
assert receipt['status']=='complete' and receipt['step']==step,receipt
assert receipt['sha256']==observed,(receipt,observed)
payload=torch.load(sys.argv[3],map_location='cpu',weights_only=False,mmap=True)
config=payload['config']; history=payload['history']
assert payload['step']==step and len(history)==step,(payload['step'],len(history))
assert config['distributed'] is False and config['distributed_world_size']==1,config
assert config['planned_optimizer_steps']==1250 and config['gradient_accumulation_steps']==4,config
assert config['full_micro_steps']==5000 and config['state_token_count']==8,config
assert config['state_architecture']=='layerwise',config
assert config['final_inference_route']=='strict_state_only',config
counts=collections.Counter(float(row['history_dropout']) for row in history)
assert counts=={0.0:1000,0.25:250},counts
out={'stage':'state8_strong1_single_step1250_artifact_audit_v1','status':'passed',
 'step':step,'sha256':observed,'state_shape':[64,8,5120],
 'history_dropout_counts':{str(k):v for k,v in sorted(counts.items())},
 'raw_history_visible_to_decoder':False,'diagnostic_only':True}
path=pathlib.Path(sys.argv[4]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY

pair_manifest="$state/c4_pair_manifest_n50.json"
if [[ ! -s "$pair_manifest" ]]; then
  "$python_bin" - "$parent_manifest" "$pair_manifest" <<'PY'
import hashlib,json,pathlib,sys
source=pathlib.Path(sys.argv[1]); value=json.loads(source.read_text())
assert value['num_pairs']==150 and len(value['pairs'])==150,value
out={**value,'num_pairs':50,'pairs':value['pairs'][:50],'diagnostic_only':True,
 'selection':'ordered_prefix','parent_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
path=pathlib.Path(sys.argv[2]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
fi

if [[ ! -s "$state/c4/recurrent_causal_gate.json" ]]; then
  exec 10>"$project/results/local_gpu_readonly_preflight.lock"
  while ! flock -n 10; do heartbeat; sleep 30; done
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
    heartbeat; sleep 30
  done
  (while true; do heartbeat; sleep 30; done) & heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --jsonl "$c4_jsonl" \
      --pair-manifest "$pair_manifest" \
      --output-json "$state/c4/recurrent_causal_gate.json" \
      --state-token-alias '<|vision_pad|>' --max-length 1536 \
      --num-pairs 50 --seed 42 --progress-dir "$state/c4/progress" \
      --progress-key strong1-single-step1250-n50 > "$state/c4/live.log" 2>&1 \
    || { flock -u 10; fail "local C4 evaluation failed"; }
  kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""; flock -u 10
fi

"$python_bin" - "$state/c4/recurrent_causal_gate.json" \
  "$state/artifact_audit.json" "$state/summary.json" <<'PY' \
  || fail "C4 summary failed"
import json,pathlib,sys
x=json.load(open(sys.argv[1])); audit=json.load(open(sys.argv[2])); s=x['summary']; c=s['clean']
assert x['artifact_sha256']==audit['sha256'],(x,audit)
assert x['state_shape']==[64,8,5120] and x['raw_history_visible_to_decoder'] is False,x
assert s['all_pairs_n']==50 and s['clean_n']>0,s
out={'stage':'state8_strong1_single_step1250_c4_v1','step':1250,
 'clean_n':s['clean_n'],'swap':c['swap']['flip_to_A_rate'],
 'random':c['random']['flip_to_A_rate'],'same_rule':c['samemal']['flip_to_A_rate'],
 'noop':c['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
 'scientific_gate_pass':s['scientific_gate_pass'],'checks':s['checks'],
 'artifact_sha256':x['artifact_sha256'],
 'history_dropout_counts':audit['history_dropout_counts'],'diagnostic_only':True,
 'excluded_from_promotion':True}
path=pathlib.Path(sys.argv[3]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
print(json.dumps(out,sort_keys=True))
PY

rm -f "$checkpoint/recurrent_state.pt" "$checkpoint/adapter/adapter_model.safetensors"
touch "$state/complete"; heartbeat
log "diagnostic step-1250 C4 complete"
