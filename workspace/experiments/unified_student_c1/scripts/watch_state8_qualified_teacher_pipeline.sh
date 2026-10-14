#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_qualified_teacher_pipeline_20260805"
train_experiment=state8-full-history-natural-teacher-recovery2-20260805
eval_experiment=state8-full-history-teacher-eval-20260805
stage_b_experiment=state8-qualified-teacher-curriculum-20260805
max_recovery_waves="${MAX_RECOVERY_WAVES:-12}"
account=csnewsandfeeds4150361735
container=amulet
teacher_prefix=studentmodeling/diagnostics/state8_full_history_natural_teacher_v1
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
status(){ timeout 90 amlt status "$1" 2>&1; }
is_pass(){ "$python_bin" - "$1" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
}
is_active(){ "$python_bin" - "$1" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_active_job
raise SystemExit(0 if has_active_job(sys.argv[1]) else 1)
PY
}
wait_experiment(){
  local experiment="$1" value
  while true; do
    heartbeat
    value="$(status "$experiment")" || { sleep 60; continue; }
    printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$value" >> "$state/status.log"
    if is_pass "$value"; then return 0; fi
    if is_active "$value"; then sleep 60; continue; fi
    amlt logs view -n 1000 "$experiment" > "$state/${experiment}.failure.log" 2>&1 || true
    return 1
  done
}
submit_if_missing(){
  local experiment="$1" config="$2" selector="$3" description="$4" required_units="$5" current rc class
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
  )" || return 1
  if [[ "$class" == missing ]]; then
    wait_quota "$required_units"
    amlt run "$config" "$selector" "$experiment" -t Feeds -y --description "$description" \
      > "$state/${experiment}.submission.txt" 2>&1 || return 1
  elif [[ "$class" != exists ]]; then
    return 1
  fi
}

wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 30; continue; }
    if "$python_bin" - "$quota" "$required" "$state/quota.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); required=int(sys.argv[2])
q=row['quotas_(used/limit)']; standard_limit=int(q['standard']['limit'])
standard=standard_limit-int(q['standard']['used'])
user_limit=int(q['user_max']['limit']); user=user_limit-int(q['user_max'].get('used') or 0)
open(sys.argv[3],'w').write(json.dumps({'selected_row':row,'required':required,
 'standard_available':standard,'user_available':user,
 'scheduler_queue_allowed':True},indent=2,sort_keys=True)+'\n')
assert standard_limit>=required and user_limit>=required
PY
    then return 0; fi
    sleep 30
  done
}

run_recoverable_stage(){
  local base="$1" recovery_prefix="$2" config="$3" selector="$4" description="$5" required_units="$6"
  local wave=0 current="$base"
  while true; do
    printf '%s\n' "$current" > "$state/active_experiment.txt"
    submit_if_missing "$current" "$config" "$selector" "$description; recovery_wave=$wave" "$required_units" \
      || return 1
    if wait_experiment "$current"; then
      LAST_EXPERIMENT="$current"
      return 0
    fi
    (( wave < max_recovery_waves )) || return 1
    wave=$((wave+1))
    current="${recovery_prefix}${wave}-20260805"
    log "recovering failed stage selector=$selector wave=$wave experiment=$current"
  done
}

"$python_bin" scripts/validate_state8_teacher_stage_a_amlt.py \
  --config amlt/feeds_state8_teacher_stage_a_20260805.yaml >> "$state/controller.log" 2>&1 \
  || fail "Stage-A teacher source config invalid"
run_recoverable_stage "$train_experiment" state8-full-history-natural-teacher-resumable-recovery \
  amlt/feeds_state8_teacher_stage_a_20260805.yaml \
  :state8-full-history-natural-teacher-seed42 \
  "Full-history natural T3 teacher with exact-epoch resumable checkpoints" 1 \
  || fail "Stage-A teacher training exhausted recovery waves"
train_experiment="$LAST_EXPERIMENT"
touch "$state/stage_a_train_pass"
if [[ ! -f "$state/teacher_published" ]]; then
  rm -rf "$state/train_download"; mkdir -p "$state/train_download"
  amlt results download "$train_experiment" --status pass \
    --include 'adapter_model.safetensors' --include 'adapter_config.json' \
    --include 'tokenizer.json' --include 'tokenizer_config.json' \
    --include 'special_tokens_map.json' --include 'added_tokens.json' \
    --include 'merges.txt' --include 'vocab.json' --include 'run_config.json' \
    --include 'teacher_view_receipt.json' --include 'teacher_eval.jsonl' \
    --output "$state/train_download" >> "$state/controller.log" 2>&1 \
    || fail "Stage-A teacher result download failed"
  readarray -t adapters < <(find "$state/train_download" -type f -path '*/model/adapter_model.safetensors')
  readarray -t configs < <(find "$state/train_download" -type f -path '*/model/adapter_config.json')
  readarray -t evals < <(find "$state/train_download" -type f -path '*/metrics/teacher_eval.jsonl')
  readarray -t views < <(find "$state/train_download" -type f -path '*/metrics/teacher_view_receipt.json')
  [[ ${#adapters[@]} -eq 1 && ${#configs[@]} -eq 1 && ${#evals[@]} -eq 1 && ${#views[@]} -eq 1 ]] \
    || fail "Stage-A teacher artifact cardinality drift"
  model_dir="$(dirname "${adapters[0]}")"
  receipt="$state/teacher_publish_receipt.json"
  "$python_bin" - "${adapters[0]}" "${evals[0]}" "${views[0]}" "$receipt" <<'PY' \
    || fail "Stage-A teacher artifact validation failed"
import hashlib,json,pathlib,sys
adapter,evaluation,view,out=map(pathlib.Path,sys.argv[1:])
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
v=json.load(open(view)); o=v['outputs']
assert v['stage']=='state8_full_history_teacher_views_v1',v
assert v['example_id_overlaps']=={'train_valid':0,'train_eval':0,'valid_eval':0},v
assert o['train']['rows']==6476 and o['valid']['rows']==300 and o['eval']['rows']==1494,v
receipt={'status':'complete','stage':'state8_full_history_teacher_publish_v1',
 'adapter_sha256':digest(adapter),'eval_sha256':digest(evaluation),
 'teacher_history_visible':True,'student_history_visible_at_final_inference':False,
 'output_schema':'label_diagnosis_evidence_v1','train_rows':6476,'valid_rows':300,'eval_rows':1494}
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
  az storage blob upload-batch --account-name "$account" --auth-mode login \
    --destination "$container" --destination-path "$teacher_prefix/model" \
    --source "$model_dir" --overwrite true >> "$state/controller.log" 2>&1 \
    || fail "Teacher model publication failed"
  az storage blob upload --account-name "$account" --auth-mode login --container-name "$container" \
    --name "$teacher_prefix/metrics/teacher_eval.jsonl" --file "${evals[0]}" --overwrite true \
    >> "$state/controller.log" 2>&1 || fail "Teacher eval manifest publication failed"
  az storage blob upload --account-name "$account" --auth-mode login --container-name "$container" \
    --name "$teacher_prefix/metrics/teacher_view_receipt.json" --file "${views[0]}" --overwrite true \
    >> "$state/controller.log" 2>&1 || fail "Teacher view receipt publication failed"
  az storage blob upload --account-name "$account" --auth-mode login --container-name "$container" \
    --name "$teacher_prefix/metrics/publish_receipt.json" --file "$receipt" --overwrite true \
    >> "$state/controller.log" 2>&1 || fail "Teacher completion receipt publication failed"
  touch "$state/teacher_published"
fi

"$python_bin" scripts/validate_state8_teacher_eval_amlt.py \
  --config amlt/feeds_state8_teacher_eval_20260805.yaml >> "$state/controller.log" 2>&1 \
  || fail "Teacher eval source config invalid"
run_recoverable_stage "$eval_experiment" state8-full-history-teacher-eval-recovery \
  amlt/feeds_state8_teacher_eval_20260805.yaml \
  :state8-full-history-teacher-eval \
  "Exact-path full-history teacher generation on frozen 1494-row T3 panel" 1 \
  || fail "Teacher evaluation exhausted recovery waves"
eval_experiment="$LAST_EXPERIMENT"
touch "$state/stage_a_eval_pass"

if [[ ! -f "$state/teacher_gate_pass" ]]; then
  rm -rf "$state/eval_download"; mkdir -p "$state/eval_download"
  amlt results download "$eval_experiment" --status pass \
    --include 'predictions.jsonl' --include 'metrics.json' --include 'exact_path_receipt.json' \
    --output "$state/eval_download" >> "$state/controller.log" 2>&1 \
    || fail "Teacher predictions download failed"
  readarray -t predictions < <(find "$state/eval_download" -type f -path '*/eval/t3/predictions.jsonl')
  [[ ${#predictions[@]} -eq 1 ]] || fail "Teacher predictions cardinality drift"
  readarray -t evals < <(find "$state/train_download" -type f -path '*/metrics/teacher_eval.jsonl')
  [[ ${#evals[@]} -eq 1 ]] || fail "Teacher eval manifest missing"
  mkdir -p "$state/semantic"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${predictions[0]}" --eval-jsonl "${evals[0]}" \
    --output-dir "$state/semantic" --judge-backend trapi --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
    >> "$state/semantic.log" 2>&1 || fail "TRAPI teacher semantic judgment failed"
  "$python_bin" scripts/validate_state8_teacher_gate.py \
    --metrics "$state/semantic/metrics.json" --output "$state/teacher_gate.json" \
    >> "$state/controller.log" 2>&1 || fail "Qualified teacher semantic gate failed"
  az storage blob upload --account-name "$account" --auth-mode login --container-name "$container" \
    --name "$teacher_prefix/metrics/teacher_gate.json" --file "$state/teacher_gate.json" --overwrite true \
    >> "$state/controller.log" 2>&1 || fail "Teacher gate publication failed"
  touch "$state/teacher_gate_pass"
fi

"$python_bin" scripts/validate_state8_qualified_teacher_curriculum_amlt.py \
  --config amlt/feeds_state8_qualified_teacher_curriculum_20260805.yaml \
  >> "$state/controller.log" 2>&1 || fail "Stage-B source config invalid"
run_recoverable_stage "$stage_b_experiment" state8-qualified-teacher-curriculum-recovery \
  amlt/feeds_state8_qualified_teacher_curriculum_20260805.yaml \
  :state8-qualified-teacher-curriculum-seed42 \
  "Stage B qualified frozen full-history teacher to eight state tokens; 5000 train-only rows; five-phase dropout; state-only inference; C4 specificity rehearsal" 4 \
  || fail "Stage-B training exhausted recovery waves"
stage_b_experiment="$LAST_EXPERIMENT"
touch "$state/stage_b_submitted"
rm -rf "$state/stage_b_download"; mkdir -p "$state/stage_b_download"
amlt results download "$stage_b_experiment" --status pass \
  --include 'adapter_model.safetensors' --include 'adapter_config.json' \
  --include 'recurrent_state.pt' --include 'completion.json' \
  --include 'qualified_teacher_receipt.json' --include 'immutable_receipt.json' \
  --include 'metrics.json' --include 'predictions.jsonl' \
  --output "$state/stage_b_download" >> "$state/controller.log" 2>&1 \
  || fail "Stage-B result download failed"
readarray -t stage_b_receipts < <(find "$state/stage_b_download" -type f -name qualified_teacher_receipt.json)
readarray -t stage_b_screens < <(find "$state/stage_b_download" -type f -path '*/eval/formal_screen/metrics.json')
[[ ${#stage_b_receipts[@]} -eq 1 && ${#stage_b_screens[@]} -eq 1 ]] \
  || fail "Stage-B result cardinality drift"
touch "$state/complete"
heartbeat
log "qualified teacher to state8 curriculum pipeline complete"
