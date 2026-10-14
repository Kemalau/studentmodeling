#!/usr/bin/env bash
set -euo pipefail

# Read-only natural-language diagnosis-C4 curve for the live one-A100 State8
# bridge.  This separates updater causality from decoder/readout causality:
# answer-C4 may move before the generated diagnosis follows the swapped state.
# The watcher never writes to the training namespace or controls training.
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_strong1_widen8_bridge_20260808/single_milestone_diagnosis_c4"
remote_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4
account=csnewsandfeeds4150361735
container=amulet
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
c4_jsonl="$project/data/processed/malrule_c1_clean/test_same_template.jsonl"
pair_manifest="$project/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
steps=(1000 2000 3000 4000 5000)
num_pairs=50

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
test -f "$pair_manifest" || fail "frozen C4 manifest missing"
test -x "$judge_python" || fail "TRAPI judge environment unavailable"
test "$(sha256sum "$pair_manifest" | cut -d' ' -f1)" = \
  edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
heartbeat

for step in "${steps[@]}"; do
  root="$state/step-$step"
  checkpoint="$root/checkpoint-$step"
  remote="$remote_root/checkpoint-$step"
  receipt="$checkpoint/checkpoint_completion.json"
  generation="$root/generation"
  semantic="$root/semantic"
  mkdir -p "$checkpoint/adapter" "$generation" "$semantic"

  if [[ -f "$root/complete" && -s "$root/summary.json" \
        && -s "$root/artifact_audit.json" ]]; then
    "$python_bin" - "$root/summary.json" "$root/artifact_audit.json" "$step" <<'PY' \
      || fail "completed diagnosis-C4 milestone binding drift step=$step"
import json,sys
s=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); step=int(sys.argv[3])
assert s['checkpoint_step']==step,s
assert s['generation']['artifact_sha256']==a['sha256'],(s,a)
assert s['generation']['raw_history_visible_to_decoder'] is False,s
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
assert config['state_architecture']=='layerwise',config
assert config['final_inference_route']=='strict_state_only',config
assert config['taxonomy_label_contract']=='source_aware_no_answer_labels_v1',config
counts={str(x):sum(float(row['history_dropout'])==x for row in history)
        for x in (0.0,0.25,0.5,0.75,1.0)}
out={'stage':'state8_strong1_single_milestone_diagnosis_c4_artifact_audit_v1',
 'status':'passed','step':step,'sha256':observed,'state_shape':[64,8,5120],
 'history_dropout_counts':counts,'raw_history_visible_to_decoder':False,
 'persistent_history_kv_cache':False}
path=pathlib.Path(sys.argv[4]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY

  if [[ ! -s "$generation/metrics.json" ]]; then
    exec 10>"$project/results/local_gpu_readonly_preflight.lock"
    while ! flock -n 10; do heartbeat; sleep 30; done
    # The measured diagnosis evaluator uses about 32 GiB.  Permit unrelated
    # small GPU work while preserving 48 GiB of free memory before launch; the
    # shared lock continues to prevent our State8 evaluators from overlapping.
    min_free_mib=49152
    while true; do
      free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
      [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= min_free_mib )) && break
      heartbeat; sleep 30
    done
    (while true; do heartbeat; sleep 30; done) & heartbeat_pid=$!
    readarray -t artifact_sha < <("$python_bin" - "$root/artifact_audit.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['sha256']['adapter_model']); print(x['sha256']['recurrent_state'])
PY
    )
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -u scripts/evaluate_recurrent_diagnosis_c4_taxonomy.py \
        --model "$model" --adapter "$checkpoint/adapter" \
        --recurrent-state "$checkpoint/recurrent_state.pt" \
        --state-architecture layerwise --jsonl "$c4_jsonl" \
        --pair-manifest "$pair_manifest" --output-dir "$generation" \
        --state-token-alias '<|vision_pad|>' --max-length 1536 \
        --max-new-tokens 160 --num-pairs "$num_pairs" --seed 42 \
        --expected-adapter-sha256 "${artifact_sha[0]}" \
        --expected-recurrent-state-sha256 "${artifact_sha[1]}" \
        > "$generation/live.log" 2>&1 \
      || {
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
        heartbeat_pid=""; flock -u 10
        fail "diagnosis-C4 generation failed step=$step"
      }
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    heartbeat_pid=""; flock -u 10
  fi

  "$python_bin" - "$generation/metrics.json" "$root/artifact_audit.json" "$num_pairs" <<'PY' \
    || fail "diagnosis-C4 generation audit failed step=$step"
import json,sys
x=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); n=int(sys.argv[3])
assert x['stage']=='state8_closed_diagnosis_c4_smoke_v1',x
assert x['num_pairs']==n and x['pair_seed']==42,x
assert x['state_shape']==[64,8,5120] and x['artifact_sha256']==a['sha256'],(x,a)
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['semantic_judgment_pending'] is True,x
PY

  if [[ ! -s "$semantic/metrics.json" ]]; then
    (while true; do heartbeat; sleep 30; done) & heartbeat_pid=$!
    PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$generation/judge_predictions.jsonl" \
      --eval-jsonl "$generation/judge_eval.jsonl" --output-dir "$semantic" \
      --judge-backend trapi --judge-workers 12 --judge-max-retries 12 \
      --judge-timeout-seconds 60 --judge-cache "$semantic/cache.jsonl" \
      --judge-audit-samples 100 --seed 42 > "$semantic/judge.log" 2>&1 \
      || {
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
        heartbeat_pid=""; fail "diagnosis-C4 semantic judge failed step=$step"
      }
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    heartbeat_pid=""
  fi

  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$semantic/predictions_redacted.jsonl" \
    --generation-metrics "$generation/metrics.json" \
    --output "$root/summary.json" > "$root/summarize.log" 2>&1 \
    || fail "diagnosis-C4 semantic summary failed step=$step"
  "$python_bin" - "$root/summary.json" "$root/artifact_audit.json" "$step" <<'PY' \
    || fail "diagnosis-C4 summary binding failed step=$step"
import json,pathlib,sys
x=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); step=int(sys.argv[3])
assert x['generation']['artifact_sha256']==a['sha256'],(x,a)
assert x['generation']['num_pairs']==50,x
x['checkpoint_step']=step
x['history_dropout_counts']=a['history_dropout_counts']
x['diagnostic_only']=True
p=pathlib.Path(sys.argv[1]); q=p.with_suffix('.json.partial')
q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); q.replace(p)
print(json.dumps({'step':step,'pass':x['pass'],
 'D_specific':x['semantic_D_specific'],
 'swap_to_donor':x['cells']['swap_to_donor']['semantic_equivalence_rate'],
 'same_rule_retention':x['same_rule_recipient_retention'],
 'noop_retention':x['noop_recipient_retention'],
 'random_stable':x['random_matches_either_stable_target_rate']},sort_keys=True))
PY

  remote_state="$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$remote/recurrent_state.pt" \
    --query exists -o tsv 2>/dev/null || true)"
  [[ "$remote_state" == true ]] || fail "remote checkpoint vanished step=$step"
  rm -f "$checkpoint/recurrent_state.pt" "$checkpoint/adapter/adapter_model.safetensors"
  touch "$root/complete"; heartbeat
  log "diagnosis-C4 milestone complete step=$step"
done

"$python_bin" - "$state" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=[]
for path in sorted(root.glob('step-*/summary.json'),key=lambda p:int(p.parent.name.split('-')[-1])):
 rows.append(json.loads(path.read_text()))
assert [row['checkpoint_step'] for row in rows]==[1000,2000,3000,4000,5000],rows
(root/'curve.json').write_text(json.dumps({
 'stage':'state8_strong1_single_milestone_diagnosis_c4_curve_v1',
 'status':'complete','semantic_judge_authoritative':True,
 'exact_label_role':'auxiliary_only_not_used_for_gate','rows':rows},
 indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat
log "diagnosis-C4 curriculum curve complete"
