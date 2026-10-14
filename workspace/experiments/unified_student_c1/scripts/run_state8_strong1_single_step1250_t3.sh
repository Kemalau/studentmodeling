#!/usr/bin/env bash
set -euo pipefail

# Read-only strict-state-only T3 diagnostic at the first 25%-dropout mixed
# checkpoint.  This is excluded from promotion and never mutates training.
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
step=1250
state="$project/results/state8_strong1_widen8_bridge_20260808/single_step1250_t3"
source_state="$project/results/state8_strong1_widen8_bridge_20260808/single_step1250_c4"
checkpoint="$state/checkpoint-$step"
remote_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4
remote="$remote_root/checkpoint-$step"
account=csnewsandfeeds4150361735
container=amulet
formal="$state/formal"
semantic="$state/semantic"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
rubric=t3-semantic-equivalence-v2-substantive-schema

mkdir -p "$checkpoint/adapter" "$formal" "$semantic"
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
test -f "$eval_jsonl" || fail "frozen formal evaluation missing"
test -x "$judge_python" || fail "TRAPI judge environment unavailable"
heartbeat

receipt="$checkpoint/checkpoint_completion.json"
if [[ ! -s "$receipt" ]]; then
  if [[ -s "$source_state/checkpoint-$step/checkpoint_completion.json" ]]; then
    cp "$source_state/checkpoint-$step/checkpoint_completion.json" "$receipt"
  else
    download_atomic "$remote/checkpoint_completion.json" "$receipt" \
      || fail "checkpoint receipt download failed"
  fi
fi
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
out={'stage':'state8_strong1_single_step1250_t3_artifact_audit_v1',
 'status':'passed','step':step,'sha256':observed,'state_shape':[64,8,5120],
 'history_dropout_counts':{str(k):v for k,v in sorted(counts.items())},
 'raw_history_visible_to_decoder':False,'diagnostic_only':True}
path=pathlib.Path(sys.argv[4]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY

if [[ ! -s "$formal/metrics.json" ]]; then
  exec 10>"$project/results/local_gpu_readonly_preflight.lock"
  while ! flock -n 10; do heartbeat; sleep 30; done
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
    heartbeat; sleep 30
  done
  (while true; do heartbeat; sleep 30; done) & heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols_taxonomy.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" \
      --output-dir "$formal" --state-token-alias '<|vision_pad|>' \
      --max-samples-per-group 25 --max-length 1536 --max-new-tokens 128 \
      --state-batch-size 2 --candidate-row-batch-size 2 --seed 42 \
      --tasks diagnose_misconception --sources eedi_mining malrule \
      --save-unredacted-predictions --t3-natural-schema \
      > "$formal/live.log" 2>&1 \
    || { flock -u 10; fail "local T3 generation failed"; }
  kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""; flock -u 10
fi

"$python_bin" - "$formal/metrics.json" "$state/artifact_audit.json" <<'PY' \
  || fail "strict T3 generation audit failed"
import json,sys
x=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2]))
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False and x['sampled_rows']==75,x
assert x['artifact_sha256']==a['sha256'],(x,a)
PY

if [[ ! -s "$semantic/metrics.json" ]]; then
  (while true; do heartbeat; sleep 30; done) & heartbeat_pid=$!
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$formal/predictions.jsonl" --eval-jsonl "$eval_jsonl" \
    --output-dir "$semantic" --judge-backend trapi --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 --max-records 75 \
    --judge-cache "$semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
    > "$semantic/judge.log" 2>&1 \
    || fail "TRAPI semantic judge failed"
  kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
fi

"$python_bin" - "$formal/metrics.json" "$semantic/metrics.json" \
  "$state/artifact_audit.json" "$state/summary.json" "$step" "$rubric" <<'PY' \
  || fail "T3 semantic summary failed"
import json,pathlib,sys
formal=json.load(open(sys.argv[1])); semantic=json.load(open(sys.argv[2]))
audit=json.load(open(sys.argv[3])); step=int(sys.argv[5]); rubric=sys.argv[6]
groups=semantic['metrics_by_group']; support=sum(int(x['llm_judge_n']) for x in groups.values())
accuracy=sum(float(x['llm_judge_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
raw_accuracy=sum(float(x['llm_judge_raw_semantic_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
schema_n=sum(int(x['llm_judge_schema_valid_n']) for x in groups.values())
assert semantic['judge_backend']=='trapi' and semantic['judge_errors']==0,semantic
assert semantic['judge_rubric']==rubric,semantic
assert support==75 and all(int(x['llm_judge_exact_shortcuts'])==0 for x in groups.values()),groups
assert formal['artifact_sha256']==audit['sha256'],(formal,audit)
out={'stage':'state8_strong1_single_step1250_t3_semantic_v2','checkpoint_step':step,
 'support':support,'semantic_accuracy':accuracy,'raw_semantic_accuracy':raw_accuracy,
 'substantive_schema_valid_rate':schema_n/support,'metrics_by_group':groups,
 'judge_rubric':rubric,'state_shape':[64,8,5120],
 'raw_history_visible_to_decoder':False,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'artifact_sha256':formal['artifact_sha256'],
 'history_dropout_counts':audit['history_dropout_counts'],
 'diagnostic_only':True,'excluded_from_promotion':True}
p=pathlib.Path(sys.argv[4]); q=p.with_suffix('.json.partial')
q.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); q.replace(p)
print(json.dumps(out,sort_keys=True))
PY

rm -f "$checkpoint/recurrent_state.pt" "$checkpoint/adapter/adapter_model.safetensors"
touch "$state/complete"; heartbeat
log "diagnostic step-1250 strict T3 complete"
