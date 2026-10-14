#!/usr/bin/env bash
set -euo pipefail

# Read-only strict-state-only T3 curve for the live one-A100 State8 bridge.
# Each result is bound to an atomic remote checkpoint and judged with the
# substantive-schema TRAPI rubric.  This watcher never controls training.
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_strong1_widen8_bridge_20260808/single_milestone_t3"
remote_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4
account=csnewsandfeeds4150361735
container=amulet
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
steps=(1000 2000 3000 4000 5000)
rubric=t3-semantic-equivalence-v2-substantive-schema

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
test -f "$eval_jsonl" || fail "frozen formal evaluation missing"
test -x "$judge_python" || fail "TRAPI judge environment unavailable"
heartbeat

for step in "${steps[@]}"; do
  root="$state/step-$step"
  checkpoint="$root/checkpoint-$step"
  receipt="$checkpoint/checkpoint_completion.json"
  remote="$remote_root/checkpoint-$step"
  formal="$root/formal"
  semantic="$root/semantic"
  mkdir -p "$checkpoint/adapter" "$formal" "$semantic"

  if [[ -f "$root/complete" && -s "$root/summary.json" \
        && -s "$root/artifact_audit.json" ]]; then
    "$python_bin" - "$root/summary.json" "$root/artifact_audit.json" "$step" <<'PY' \
      || fail "completed T3 milestone binding drift step=$step"
import json,sys
s=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); step=int(sys.argv[3])
assert s['checkpoint_step']==step and s['artifact_sha256']==a['sha256'],(s,a)
assert s['raw_history_visible_to_decoder'] is False and s['diagnostic_only'] is True,s
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
assert config['state_architecture']=='layerwise',config
assert config['final_inference_route']=='strict_state_only',config
counts={str(x):sum(float(row['history_dropout'])==x for row in history)
        for x in (0.0,0.25,0.5,0.75,1.0)}
out={'stage':'state8_strong1_single_milestone_t3_artifact_audit_v1',
     'status':'passed','step':step,'sha256':observed,'state_shape':[64,8,5120],
     'history_dropout_counts':counts,'raw_history_visible_to_decoder':False}
path=pathlib.Path(sys.argv[4]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY

  if [[ ! -s "$formal/metrics.json" ]]; then
    exec 10>"$project/results/local_gpu_readonly_preflight.lock"
    while ! flock -n 10; do heartbeat; sleep 30; done
    # This evaluator peaks near 32 GiB on the local 80 GiB A100.  Requiring an
    # entirely idle device unnecessarily blocks behind unrelated small
    # forward-only jobs.  The shared lock still serializes all of our State8
    # evaluators; require 48 GiB free so this job keeps at least ~16 GiB of
    # measured safety headroom at its observed peak.
    min_free_mib=49152
    while true; do
      free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
      [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= min_free_mib )) && break
      heartbeat; sleep 30
    done
    (while true; do heartbeat; sleep 30; done) &
    heartbeat_pid=$!
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
      || {
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
        heartbeat_pid=""; flock -u 10; fail "local T3 generation failed step=$step"
      }
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    heartbeat_pid=""; flock -u 10
  fi

  "$python_bin" - "$formal/metrics.json" "$root/artifact_audit.json" <<'PY' \
    || fail "strict T3 generation audit failed step=$step"
import json,sys
x=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2]))
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False and x['sampled_rows']==75,x
assert x['artifact_sha256']==a['sha256'],(x,a)
PY

  if [[ ! -s "$semantic/metrics.json" ]]; then
    (while true; do heartbeat; sleep 30; done) &
    heartbeat_pid=$!
    PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$formal/predictions.jsonl" --eval-jsonl "$eval_jsonl" \
      --output-dir "$semantic" --judge-backend trapi --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 --max-records 75 \
      --judge-cache "$semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
      > "$semantic/judge.log" 2>&1 \
      || {
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
        heartbeat_pid=""; fail "TRAPI semantic judge failed step=$step"
      }
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    heartbeat_pid=""
  fi

  "$python_bin" - "$formal/metrics.json" "$semantic/metrics.json" \
    "$root/artifact_audit.json" "$root/summary.json" "$step" "$rubric" <<'PY' \
    || fail "T3 semantic summary failed step=$step"
import json,pathlib,sys
formal=json.load(open(sys.argv[1])); semantic=json.load(open(sys.argv[2]))
audit=json.load(open(sys.argv[3])); step=int(sys.argv[5]); rubric=sys.argv[6]
groups=semantic['metrics_by_group']
support=sum(int(row['llm_judge_n']) for row in groups.values())
accuracy=sum(float(row['llm_judge_accuracy'])*int(row['llm_judge_n'])
             for row in groups.values())/support
raw_accuracy=sum(float(row['llm_judge_raw_semantic_accuracy'])*int(row['llm_judge_n'])
                 for row in groups.values())/support
schema_n=sum(int(row['llm_judge_schema_valid_n']) for row in groups.values())
assert semantic['judge_backend']=='trapi' and semantic['judge_errors']==0,semantic
assert semantic['judge_rubric']==rubric,semantic
assert support==75 and all(int(row['llm_judge_exact_shortcuts'])==0
                           for row in groups.values()),groups
assert formal['artifact_sha256']==audit['sha256'],(formal,audit)
out={'stage':'state8_strong1_single_milestone_t3_semantic_v2',
 'checkpoint_step':step,'support':support,'semantic_accuracy':accuracy,
 'raw_semantic_accuracy':raw_accuracy,'substantive_schema_valid_rate':schema_n/support,
 'metrics_by_group':groups,'judge_rubric':rubric,'state_shape':[64,8,5120],
 'raw_history_visible_to_decoder':False,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'artifact_sha256':formal['artifact_sha256'],
 'history_dropout_counts':audit['history_dropout_counts'],'diagnostic_only':True}
p=pathlib.Path(sys.argv[4]); q=p.with_suffix('.json.partial')
q.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); q.replace(p)
print(json.dumps(out,sort_keys=True))
PY

  remote_state="$(az storage blob exists --auth-mode login \
    --account-name "$account" --container-name "$container" \
    --name "$remote/recurrent_state.pt" --query exists -o tsv 2>/dev/null || true)"
  [[ "$remote_state" == true ]] || fail "remote checkpoint vanished step=$step"
  rm -f "$checkpoint/recurrent_state.pt" \
    "$checkpoint/adapter/adapter_model.safetensors"
  touch "$root/complete"; heartbeat
  log "strict T3 milestone complete step=$step"
done

"$python_bin" - "$state" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=[]
for path in sorted(root.glob('step-*/summary.json'),key=lambda p:int(p.parent.name.split('-')[-1])):
 rows.append(json.loads(path.read_text()))
assert [row['checkpoint_step'] for row in rows]==[1000,2000,3000,4000,5000],rows
(root/'curve.json').write_text(json.dumps({
 'stage':'state8_strong1_single_milestone_t3_curve_v2',
 'status':'complete','rows':rows},indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat
log "strict T3 curriculum curve complete"
