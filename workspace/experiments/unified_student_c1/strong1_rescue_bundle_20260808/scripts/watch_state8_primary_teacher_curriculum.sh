#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
teacher_state="$project/results/state8_qualified_teacher_pipeline_20260805"
state="$teacher_state/primary_curriculum"
config="$project/amlt/feeds_state8_qualified_teacher_curriculum_20260805.yaml"
base_experiment=state8-qualified-teacher-curriculum-20260805
job=state8-qualified-teacher-curriculum-seed42
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
max_recovery_waves="${MAX_RECOVERY_WAVES:-32}"
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

while [[ ! -f "$teacher_state/teacher_gate_pass" ]]; do
  heartbeat
  sleep 60
done
test -f "$teacher_state/teacher_gate.json" || fail "published teacher gate is missing"
test -f "$teacher_state/teacher_provenance_gate.json" \
  || fail "published teacher provenance gate is missing"
"$python_bin" scripts/validate_state8_qualified_teacher_curriculum_amlt.py \
  --config "$config" >> "$state/controller.log" 2>&1 \
  || fail "primary Stage-B source config invalid"

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 30; continue; }
    if "$python_bin" - "$quota" "$state/quota.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); q=row['quotas_(used/limit)']
standard_limit=int(q['standard']['limit']); standard_used=int(q['standard']['used'])
user_limit=int(q['user_max']['limit']); user_used=int(q['user_max'].get('used') or 0)
open(sys.argv[2],'w').write(json.dumps({'selected_row':row,'required':4,
 'standard_available':standard_limit-standard_used,'user_available':user_limit-user_used,
 'scheduler_queue_allowed':True},indent=2,sort_keys=True)+'\n')
assert standard_limit>=4 and user_limit>=4
PY
    then return 0; fi
    sleep 30
  done
}

wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260805 ]] && wave="${BASH_REMATCH[1]}"
else
  experiment="$base_experiment"
fi
while true; do
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "could not classify primary Stage-B experiment"; }
  if [[ "$class" == missing ]]; then
    wait_quota
    amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "Qualified teacher T3-heavy primary; exact 5000 rows; five-phase dropout; state-only inference; C4 specificity; recovery_wave=$wave" \
      > "$state/${experiment}.submission.txt" 2>&1 \
      || { flock -u 8; fail "primary Stage-B submission failed wave=$wave"; }
  elif [[ "$class" != exists ]]; then
    flock -u 8
    fail "ambiguous primary Stage-B experiment lookup"
  fi
  flock -u 8
  while true; do
    heartbeat
    status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 60; continue; }
    printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
    then break 2; fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
    then sleep 60; continue; fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
raise SystemExit(0 if paused_job_names(sys.argv[1]) else 1)
PY
    then
      log "requesting resume for paused primary Stage-B job"
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      sleep 300; continue
    fi
    timeout 180 amlt logs view -n 2000 "$experiment" ":$job" \
      > "$state/${experiment}.failure.log" 2>&1 || true
    break
  done
  (( wave < max_recovery_waves )) || fail "primary Stage-B exhausted recovery waves"
  wave=$((wave+1))
  experiment="state8-qualified-teacher-curriculum-recovery${wave}-20260805"
done

exec 5>"$teacher_state/stage_b_finalize.lock"; flock -x 5
if [[ ! -f "$teacher_state/complete" ]]; then
  rm -rf "$teacher_state/stage_b_download"; mkdir -p "$teacher_state/stage_b_download"
  amlt results download "$experiment" --status pass \
    --include 'adapter_model.safetensors' --include 'adapter_config.json' \
    --include 'recurrent_state.pt' --include 'completion.json' \
    --include 'qualified_teacher_receipt.json' --include 'immutable_receipt.json' \
    --include 'metrics.json' --include 'predictions.jsonl' \
    --output "$teacher_state/stage_b_download" >> "$state/controller.log" 2>&1 \
    || { flock -u 5; fail "primary Stage-B result download failed"; }
  readarray -t receipts < <(find "$teacher_state/stage_b_download" -type f -name qualified_teacher_receipt.json)
  readarray -t immutable < <(find "$teacher_state/stage_b_download" -type f -name immutable_receipt.json)
  readarray -t screens < <(find "$teacher_state/stage_b_download" -type f -path '*/eval/formal_screen/metrics.json')
  [[ ${#receipts[@]} -eq 1 && ${#immutable[@]} -eq 1 && ${#screens[@]} -eq 1 ]] \
    || { flock -u 5; fail "primary Stage-B artifact cardinality drift"; }
  "$python_bin" - "${receipts[0]}" "${immutable[0]}" "${screens[0]}" <<'PY' \
    || { flock -u 5; fail "primary Stage-B receipt validation failed"; }
import json,sys
q=json.load(open(sys.argv[1])); i=json.load(open(sys.argv[2])); m=json.load(open(sys.argv[3]))
assert q['stage']=='state8_qualified_teacher_curriculum_v1' and q['replay_variant']=='t3_heavy',q
assert q['micro_steps']==5000 and q['state_shape']==[64,8,5120],q
assert q['raw_history_visible_to_decoder'] is False and q['persistent_history_kv_cache'] is False,q
assert i['stage']=='state8_qualified_teacher_curriculum_immutable_v1',i
assert i['sha256']=={'adapter_model':q['adapter_sha256'],'recurrent_state':q['recurrent_state_sha256']},(i,q)
assert m['state_shape']==[64,8,5120] and m['sampled_rows']==487,m
assert m['raw_history_visible_to_decoder'] is False and m['persistent_history_kv_cache'] is False,m
PY
  touch "$teacher_state/stage_b_submitted" "$teacher_state/complete"
fi
touch "$state/complete"
flock -u 5
heartbeat
log "independent primary curriculum controller completed Stage B"
