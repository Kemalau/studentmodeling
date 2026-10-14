#!/usr/bin/env bash
set -euo pipefail

# Read-only local C4 curve for the live one-A100 State8 bridge.  This watcher
# never writes to the training namespace and never controls the remote job.
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_strong1_widen8_bridge_20260808/single_milestone_c4"
remote_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4
account=csnewsandfeeds4150361735
container=amulet
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
c4_jsonl="$project/data/processed/malrule_c1_clean/test_same_template.jsonl"
parent_manifest="$project/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"
# Step 1000 is the final 0%-dropout row because progress is (step-1)/4999;
# steps 2000/3000/4000 similarly close the 25/50/75% phases, and step 5000
# measures the completed state-only phase.
steps=(250 1000 2000 3000 4000 5000)

mkdir -p "$state"
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

for step in "${steps[@]}"; do
  root="$state/step-$step"; checkpoint="$root/checkpoint-$step"
  receipt="$checkpoint/checkpoint_completion.json"
  remote="$remote_root/checkpoint-$step"
  mkdir -p "$checkpoint/adapter" "$root/c4/progress"

  if [[ -f "$root/complete" && -s "$root/summary.json" \
        && -s "$root/artifact_audit.json" ]]; then
    "$python_bin" - "$root/summary.json" "$root/artifact_audit.json" "$step" <<'PY' \
      || fail "completed answer-C4 milestone binding drift step=$step"
import json,sys
s=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); step=int(sys.argv[3])
assert s['step']==step and s['artifact_sha256']==a['sha256'],(s,a)
assert s['diagnostic_only'] is True,s
PY
    heartbeat; continue
  fi
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
      || fail "checkpoint download failed step=$step file=$file"
  done

  "$python_bin" - "$receipt" "$checkpoint/adapter/adapter_model.safetensors" \
    "$checkpoint/recurrent_state.pt" "$root/artifact_audit.json" "$step" <<'PY' \
    || fail "checkpoint audit failed step=$step"
import hashlib,json,pathlib,sys,torch
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
assert payload['step']==step and len(history)==step,(payload['step'],len(history),step)
assert config['distributed'] is False and config['distributed_world_size']==1,config
assert config['planned_optimizer_steps']==1250 and config['gradient_accumulation_steps']==4,config
assert config['full_micro_steps']==5000 and config['state_token_count']==8,config
assert config['state_architecture']=='layerwise' and config['final_inference_route']=='strict_state_only',config
out={'stage':'state8_strong1_single_milestone_artifact_audit_v1','status':'passed',
     'step':step,'sha256':observed,'state_shape':[64,8,5120],
     'history_dropout_counts':{str(x):sum(float(r['history_dropout'])==x for r in history)
                               for x in (0.0,0.25,0.5,0.75,1.0)},
     'raw_history_visible_to_decoder':False}
path=pathlib.Path(sys.argv[4]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY

  pair_manifest="$root/c4_pair_manifest_n50.json"
  if [[ ! -s "$pair_manifest" ]]; then
    "$python_bin" - "$parent_manifest" "$pair_manifest" <<'PY'
import hashlib,json,pathlib,sys
source=pathlib.Path(sys.argv[1]); x=json.loads(source.read_text())
assert x['num_pairs']==150 and len(x['pairs'])==150,x
out={**x,'num_pairs':50,'pairs':x['pairs'][:50],'diagnostic_only':True,
     'selection':'ordered_prefix','parent_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
path=pathlib.Path(sys.argv[2]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
  fi

  if [[ ! -s "$root/c4/recurrent_causal_gate.json" ]]; then
    exec 10>"$project/results/local_gpu_readonly_preflight.lock"
    while ! flock -n 10; do heartbeat; sleep 30; done
    # The measured answer-C4 evaluator uses about 32 GiB.  The shared lock
    # serializes our State8 jobs, so an unrelated process need not force total
    # idleness; launch only with a conservative 48 GiB free-memory floor.
    min_free_mib=49152
    while true; do
      free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
      [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= min_free_mib )) && break
      heartbeat; sleep 30
    done
    heartbeat
    (while true; do heartbeat; sleep 30; done) &
    heartbeat_pid=$!
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
        --model "$model" --adapter "$checkpoint/adapter" \
        --recurrent-state "$checkpoint/recurrent_state.pt" \
        --state-architecture layerwise --jsonl "$c4_jsonl" \
        --pair-manifest "$pair_manifest" \
        --output-json "$root/c4/recurrent_causal_gate.json" \
        --state-token-alias '<|vision_pad|>' --max-length 1536 \
        --num-pairs 50 --seed 42 --progress-dir "$root/c4/progress" \
        --progress-key "strong1-single-step${step}-n50" \
      > "$root/c4/live.log" 2>&1 \
      || {
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
        heartbeat_pid=""
        flock -u 10
        fail "local C4 failed step=$step"
      }
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    heartbeat_pid=""
    flock -u 10
  fi

  "$python_bin" - "$root/c4/recurrent_causal_gate.json" "$root/summary.json" "$step" <<'PY' \
    || fail "C4 summary failed step=$step"
import json,pathlib,sys
x=json.load(open(sys.argv[1])); s=x['summary']; c=s['clean']; step=int(sys.argv[3])
assert x['state_shape']==[64,8,5120] and x['raw_history_visible_to_decoder'] is False,x
assert s['all_pairs_n']==50 and s['clean_n']>0,s
out={'stage':'state8_strong1_single_milestone_c4_v1','step':step,
 'clean_n':s['clean_n'],'swap':c['swap']['flip_to_A_rate'],
 'random':c['random']['flip_to_A_rate'],'same_rule':c['samemal']['flip_to_A_rate'],
 'noop':c['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
 'scientific_gate_pass':s['scientific_gate_pass'],'checks':s['checks'],
 'artifact_sha256':x['artifact_sha256'],'diagnostic_only':True}
path=pathlib.Path(sys.argv[2]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
print(json.dumps(out,sort_keys=True))
PY
  touch "$root/complete"; heartbeat
  log "local milestone C4 complete step=$step"
done

"$python_bin" - "$state" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=[]
for path in sorted(root.glob('step-*/summary.json'),key=lambda p:int(p.parent.name.split('-')[-1])):
 rows.append(json.loads(path.read_text()))
assert [x['step'] for x in rows]==[250,1000,2000,3000,4000,5000],rows
(root/'curve.json').write_text(json.dumps({'stage':'state8_strong1_single_milestone_c4_curve_v1',
 'status':'complete','rows':rows},indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat
log "local milestone C4 curve complete"
