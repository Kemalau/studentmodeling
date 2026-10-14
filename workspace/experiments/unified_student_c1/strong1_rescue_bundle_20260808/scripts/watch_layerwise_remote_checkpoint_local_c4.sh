#!/usr/bin/env bash
set -euo pipefail

: "${STATE_SLUG:?STATE_SLUG is required}"
: "${LOCAL_C4_SUBDIR:?LOCAL_C4_SUBDIR is required}"
: "${CHECKPOINT_ROUND:?CHECKPOINT_ROUND is required}"
: "${REMOTE_PREFIX:?REMOTE_PREFIX must name the complete checkpoint directory}"
: "${ARTIFACT_LABEL:?ARTIFACT_LABEL is required}"

expected_world_size="${EXPECTED_WORLD_SIZE:-4}"
expected_promoted_manifest="${EXPECTED_PROMOTED_MANIFEST:-}"
expected_planned_steps="${EXPECTED_PLANNED_OPTIMIZER_STEPS:-}"

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_dir="${LOCAL_CHECKPOINT_STATE_DIR:-$project_dir/results/layerwise_unified_final_pipeline_20260731/$STATE_SLUG/$LOCAL_C4_SUBDIR}"
completion_marker="${LOCAL_C4_COMPLETION_MARKER:-$state_dir/complete}"
checkpoint="$state_dir/checkpoint-round-$CHECKPOINT_ROUND"
adapter="$checkpoint/adapter"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
account=csnewsandfeeds4150361735
container=amulet
c4_jsonl="$project_dir/data/processed/malrule_c1_clean/test_same_template.jsonl"
parent_manifest="$project_dir/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"
pair_manifest="$state_dir/c4_pair_manifest_n50.json"

mkdir -p "$state_dir" "$adapter" "$state_dir/c4/progress"
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

receipt="$checkpoint/checkpoint_completion.json"
while [[ ! -s "$receipt" ]]; do
  heartbeat
  exists="$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$REMOTE_PREFIX/checkpoint_completion.json" \
    --query exists -o tsv 2>/dev/null || true)"
  if [[ "$exists" == true ]]; then
    download_atomic "$REMOTE_PREFIX/checkpoint_completion.json" "$receipt" \
      || { sleep 120; continue; }
  else
    sleep 120
  fi
done

"$python_bin" - "$receipt" "$CHECKPOINT_ROUND" "$expected_world_size" <<'PY' \
  || fail "checkpoint completion receipt invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x.get('status')=='complete' and x.get('round')==int(sys.argv[2]),x
assert x.get('world_size')==int(sys.argv[3]),x
assert set(x.get('sha256') or {})=={'adapter_model','recurrent_state'},x
PY

if [[ ! -s "$adapter/adapter_model.safetensors" ]]; then
  download_atomic "$REMOTE_PREFIX/adapter/adapter_model.safetensors" \
    "$adapter/adapter_model.safetensors" || fail "adapter download failed"
fi
if [[ ! -s "$adapter/adapter_config.json" ]]; then
  download_atomic "$REMOTE_PREFIX/adapter/adapter_config.json" \
    "$adapter/adapter_config.json" || fail "adapter config download failed"
fi
if [[ ! -s "$checkpoint/recurrent_state.pt" ]]; then
  download_atomic "$REMOTE_PREFIX/recurrent_state.pt" "$checkpoint/recurrent_state.pt" \
    || fail "recurrent-state download failed"
fi

"$python_bin" - "$receipt" "$adapter/adapter_model.safetensors" \
  "$checkpoint/recurrent_state.pt" "$state_dir/artifact_audit.json" \
  "$ARTIFACT_LABEL" "$expected_planned_steps" <<'PY' || fail "downloaded checkpoint hash audit failed"
import hashlib,json,pathlib,sys,torch
receipt=json.load(open(sys.argv[1],encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(sys.argv[2]),'recurrent_state':digest(sys.argv[3])}
assert observed==receipt['sha256'],(observed,receipt)
expected=sys.argv[6].strip()
planned=None
if expected:
    payload=torch.load(sys.argv[3],map_location='cpu',weights_only=False)
    config=payload.get('config') or {}
    planned=config.get('planned_optimizer_steps')
    assert planned==int(expected),(planned,expected)
    assert config.get('state_architecture')=='layerwise',config
    assert config.get('state_token_count')==8,config
out={'stage':'layerwise_checkpoint_local_artifact_audit','artifact_label':sys.argv[5],
     'status':'passed','round':receipt['round'],'world_size':receipt['world_size'],
     'sha256':observed,'expected_planned_optimizer_steps':int(expected) if expected else None,
     'observed_planned_optimizer_steps':planned}
path=pathlib.Path(sys.argv[4]); tmp=path.with_suffix(path.suffix+'.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY

if [[ "$expected_world_size" == 16 ]]; then
  exact_args=(
    --checkpoint "$checkpoint/recurrent_state.pt"
    --expected-round "$CHECKPOINT_ROUND"
    --output "$state_dir/exact_resume_audit.json"
  )
  if [[ -n "$expected_promoted_manifest" ]]; then
    test -s "$expected_promoted_manifest" \
      || fail "expected promoted manifest is missing"
  fi
  "$python_bin" scripts/audit_layerwise_distributed_checkpoint.py \
    "${exact_args[@]}" >> "$state_dir/controller.log" 2>&1 \
    || fail "16-rank exact-resume/promotion audit failed"
  if [[ -n "$expected_promoted_manifest" ]]; then
    "$python_bin" scripts/audit_layerwise_promoted_checkpoint.py \
      --checkpoint "$checkpoint/recurrent_state.pt" \
      --manifest "$expected_promoted_manifest" \
      --expected-round "$CHECKPOINT_ROUND" \
      --output "$state_dir/promoted_recipe_audit.json" \
      >> "$state_dir/controller.log" 2>&1 \
      || fail "16-rank promoted-recipe audit failed"
  fi
fi

test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B missing"
test -f "$c4_jsonl" || fail "frozen C4 source missing"
test -f "$parent_manifest" || fail "frozen C4 pair manifest missing"

if [[ ! -s "$pair_manifest" ]]; then
  "$python_bin" - "$parent_manifest" "$pair_manifest" <<'PY' \
    || fail "C4 n=50 pair-manifest derivation failed"
import hashlib,json,pathlib,sys
source=pathlib.Path(sys.argv[1]); value=json.loads(source.read_text(encoding='utf-8'))
assert value.get('num_pairs')==150 and len(value.get('pairs') or [])==150,value
out={**value,'num_pairs':50,'pairs':value['pairs'][:50],'diagnostic_only':True,
     'selection':'ordered_prefix','parent_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
path=pathlib.Path(sys.argv[2]); tmp=path.with_suffix(path.suffix+'.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
fi

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 60; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat
  sleep 60
done

if [[ ! -s "$state_dir/c4/recurrent_causal_gate.json" ]]; then
  (
    while true; do heartbeat; sleep 60; done
  ) &
  heartbeat_pid=$!
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
      --progress-key "$ARTIFACT_LABEL-round${CHECKPOINT_ROUND}-local-n50" \
      >> "$state_dir/c4/live.log" 2>&1 \
    || fail "local checkpoint C4 n=50 failed"
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
fi

"$python_bin" - "$state_dir/c4/recurrent_causal_gate.json" \
  "$state_dir/summary.json" "$ARTIFACT_LABEL" "$CHECKPOINT_ROUND" <<'PY' \
  || fail "local C4 summary invalid"
import json,pathlib,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))['summary']; c=x['clean']
assert x['all_pairs_n']==50 and x['clean_n']>0,x
out={'stage':'layerwise_checkpoint_local_c4','artifact_label':sys.argv[3],
     'checkpoint_round':int(sys.argv[4]),'diagnostic_only':True,'not_a_formal_gate':True,
     'clean_n':x['clean_n'],'swap':c['swap']['flip_to_A_rate'],
     'random':c['random']['flip_to_A_rate'],'same_rule':c['samemal']['flip_to_A_rate'],
     'noop':c['noop']['flip_to_A_rate'],'D_specific':x['D_specific'],
     'scientific_gate_pass':x['scientific_gate_pass'],'checks':x['checks']}
path=pathlib.Path(sys.argv[2]); tmp=path.with_suffix(path.suffix+'.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
print(json.dumps(out,indent=2,sort_keys=True))
PY
touch "$completion_marker"
heartbeat
log "$ARTIFACT_LABEL round-$CHECKPOINT_ROUND local C4-only audit complete"
