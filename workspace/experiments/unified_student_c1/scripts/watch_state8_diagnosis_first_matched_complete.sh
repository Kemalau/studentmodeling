#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
upstream="$project/results/state8_diagnosis_first_matched_curriculum_20260810"
c3="$project/results/c3_exposure_matched_world2_20260809"
state="$project/results/state8_diagnosis_first_matched_complete_20260810"
fit="$state/final_fit"
eval_state="$state/final_eval"
fit_progress="$fit/progress"
config_world4="$project/amlt/feeds_state8_diagnosis_first_matched_final_fit_world4_20260810.yaml"
config_world2="$project/amlt/feeds_state8_diagnosis_first_matched_final_fit_world2_20260810.yaml"
eval_config="$project/amlt/feeds_state8_diagnosis_first_matched_final_eval_generated_20260810.yaml"
job=state8-diagnosis-first-matched-seed67-unified-finalfit-seed42
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
max_waves=32
fit_stall_seconds="${STATE8_MATCHED_FINALFIT_STALL_SECONDS:-7200}"
fit_startup_stall_seconds="${STATE8_MATCHED_FINALFIT_STARTUP_STALL_SECONDS:-14400}"
fit_posttrain_stall_seconds="${STATE8_MATCHED_FINALFIT_POSTTRAIN_STALL_SECONDS:-21600}"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"

mkdir -p "$state" "$fit" "$fit_progress" "$eval_state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }

while [[ ! -f "$upstream/complete" ]]; do
  heartbeat
  [[ ! -f "$upstream/needs_attention" ]] || fail "matched curriculum needs attention"
  sleep_unlocked 60
done
[[ -f "$upstream/scientific_gate_pass" ]] || {
  touch "$state/scientific_gate_failed"
  fail "matched curriculum did not pass its joint promotion gate"
}
selection="$upstream/selection.json"
"$python_bin" - "$selection" <<'PY' || fail "invalid matched promotion receipt"
import json,sys
x=json.load(open(sys.argv[1])); s=x.get('selected') or {}
assert x['stage']=='state8_diagnosis_first_matched_joint_promotion_v1',x
assert x['pass'] is True and s['qualified'] is True,x
assert s['slug']=='diagnosis_first_matched_seed67',s
assert s['finalfit_variant']=='diagnosis_first_matched_seed67',s
assert s['finalfit_job']=='state8-diagnosis-first-matched-seed67-unified-finalfit-seed42',s
assert s['checks']['answer_C4_pass'] is True,s
assert s['checks']['diagnosis_C4_pass'] is True,s
assert x['semantic_judge_authoritative'] is True,x
assert x['exact_match_authoritative'] is False,x
PY

"$python_bin" scripts/materialize_state8_diagnosis_first_matched_final_eval.py \
  --source amlt/feeds_state8_final_eval_20260805.yaml --output "$eval_config" \
  > "$state/materialize_final_eval.log" 2>&1 \
  || fail "could not materialize matched final evaluator"
"$python_bin" scripts/validate_state8_final_eval_amlt.py --config "$eval_config" \
  >> "$state/materialize_final_eval.log" 2>&1 \
  || fail "matched final evaluator failed validation"

available_slots(){
  local audit="$fit/active_a100.json"
  "$python_bin" scripts/count_active_a100_requests.py --user-limit 32 --recent 500 \
    --output "$audit" >/dev/null 2>> "$fit/controller.log" || return 1
  "$python_bin" - "$audit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['auditable'] is True,x
print(int(x['available_user_slots']))
PY
}

quota_slots(){
  local quota="$fit/quota.json"
  timeout 90 amlt --json-tables ti sing -t Feeds > "$quota" 2>> "$fit/controller.log" \
    || return 1
  "$python_bin" - "$quota" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1])); matches=[x for x in rows
 if x.get('series')=='NDAMv4' and (x.get('accelerator') or {}).get('type')=='A100'
 and int((x.get('accelerator') or {}).get('memory_gb',-1))==80]
assert len(matches)==1,matches
q=matches[0]['quotas_(used/limit)']['standard']; print(int(q['limit'])-int(q['used']))
PY
}

topology_receipt="$fit/topology_selection.json"
if [[ ! -f "$topology_receipt" ]]; then
  user_available="$(available_slots)" || fail "could not audit user A100 slots for final-fit topology"
  cluster_available="$(quota_slots)" || fail "could not audit cluster A100 slots for final-fit topology"
  "$python_bin" scripts/select_state8_finalfit_topology.py \
    --output "$topology_receipt" --topology "${FINALFIT_TOPOLOGY:-auto}" \
    --profile diagnosis_first_matched \
    --available-user-slots "$user_available" \
    --available-cluster-slots "$cluster_available" \
    > "$fit/topology_selection.stdout" 2> "$fit/topology_selection.stderr" \
    || fail "matched final-fit topology selection failed"
fi
readarray -t topology_fields < <("$python_bin" - "$topology_receipt" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['profile']=='diagnosis_first_matched',x
assert x['cross_topology_resume_allowed'] is False,x
for key in ('topology','config','validator','gpu_count','distributed_rounds'):
 print(x[key])
PY
)
[[ ${#topology_fields[@]} -eq 5 ]] || fail "invalid matched final-fit topology receipt"
fit_topology="${topology_fields[0]}"
fit_gpu_count="${topology_fields[3]}"
fit_distributed_rounds="${topology_fields[4]}"
case "$fit_topology" in
  world4) config="$config_world4" ;;
  world2) config="$config_world2" ;;
  *) fail "unsupported matched final-fit topology=$fit_topology" ;;
esac
test "$config" = "$project/${topology_fields[1]}" \
  || fail "matched pinned final-fit config path drift"
fit_validator="$project/${topology_fields[2]}"
test -f "$config" && test -f "$fit_validator" \
  || fail "matched pinned final-fit config/validator missing"
base="state8-diagnosis-first-matched-unified-finalfit-${fit_topology}-seed42-20260810"
recovery_prefix="state8-diagnosis-first-matched-unified-finalfit-recovery-${fit_topology}-"

submit_wave(){
  local experiment="$1" current rc class output user_available cluster_available
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  user_available="$(available_slots)" || { flock -u 8; return 1; }
  cluster_available="$(quota_slots)" || { flock -u 8; return 1; }
  if (( user_available < fit_gpu_count || cluster_available < fit_gpu_count )); then
    flock -u 8; return 2
  fi
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$fit/$experiment.preflight.yaml" 2> "$fit/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$fit_validator" \
    --config "$fit/$experiment.preflight.yaml" >> "$fit/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "Diagnosis-first matched State8 exact 259845-row epoch; 8 state tokens; strict state-only decoder; full-history Teacher KD; semantic-first schema; pinned ${fit_topology}" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$fit/$experiment.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$fit/active_experiment.txt"
  return 0
}

observe_finalfit_progress(){
  local experiment="$1" status_text="$2" listing log_file threshold_rc marker
  grep -Eqi '\brunning\b' <<< "$status_text" || {
    rm -f "$fit_progress/stalled.json" "$fit_progress/health_alert.json"
    return 0
  }
  listing="$(timeout 90 amlt logs list "$experiment" ":$job" 2>/dev/null)" \
    || return 0
  log_file="$(LISTING="$listing" "$python_bin" - <<'PY'
import os,re
paths=[]
for line in os.environ['LISTING'].splitlines():
 value=line.strip()
 if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
  retry=re.search(r'retry_(\d+)',value)
  paths.append((int(retry.group(1)) if retry else -1,value))
print(max(paths)[1] if paths else '')
PY
)" || return 0
  [[ -n "$log_file" ]] || return 0
  timeout 90 amlt logs view -F "$log_file" -n 900 "$experiment" ":$job" \
    > "$fit_progress/latest_rank0.log" 2> "$fit_progress/latest_rank0.stderr" \
    || return 0
  "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
    --log "$fit_progress/latest_rank0.log" --experiment "$experiment" \
    --log-file "$log_file" --status running --output "$fit_progress/progress.json" \
    --total-rounds "$fit_distributed_rounds" --max-run-seconds 43200 \
    --validation-reserve-seconds 21600 >> "$fit/controller.log" 2>&1 \
    || return 0
  "$python_bin" - "$fit_progress/progress.json" "$fit_stall_seconds" \
    "$fit_startup_stall_seconds" "$fit_posttrain_stall_seconds" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); health=x.get('health') or {}
unhealthy=health.get('healthy') is False
complete=(float(x.get('progress_fraction') or 0)>=1.0
          or int(x.get('round') or -1)>=int(x.get('total_rounds') or 16241))
awaiting=x.get('phase')=='awaiting_first_training_record'
threshold=float(sys.argv[4] if complete else sys.argv[3] if awaiting else sys.argv[2])
stalled=float(x.get('seconds_since_progress') or 0)>=threshold
raise SystemExit(0 if stalled or unhealthy else 1)
PY
  threshold_rc=$?
  if (( threshold_rc != 0 )); then
    rm -f "$fit_progress/stalled.json" "$fit_progress/health_alert.json"
    return 0
  fi
  cp "$fit_progress/progress.json" "$fit_progress/stalled.json"
  marker="$fit/$experiment.stall_cancel_requested"
  [[ ! -f "$marker" ]] || return 2
  timeout 90 amlt show "$experiment" > "$fit/$experiment.stalled.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
    > "$fit/$experiment.stalled.log" 2>&1 || true
  if timeout 90 amlt cancel "$experiment" ":$job" -y \
      > "$fit/$experiment.stalled.cancel.txt" 2>&1; then
    touch "$marker"
    log "cancelled stalled/unhealthy matched final fit for atomic-checkpoint recovery experiment=$experiment"
    return 2
  fi
  return 0
}

if [[ ! -f "$fit/complete" ]]; then
  wave=0
  if [[ -s "$fit/active_experiment.txt" ]]; then
    experiment="$(<"$fit/active_experiment.txt")"
    [[ "$experiment" =~ recovery-${fit_topology}-([0-9]+)-20260810$ ]] && wave="${BASH_REMATCH[1]}"
  else
    experiment="$base"
    while ! submit_wave "$experiment"; do
      heartbeat; log "waiting for $fit_gpu_count auditable A100 slots for matched final fit topology=$fit_topology"; sleep_unlocked 120
    done
    log "submitted matched exact final fit experiment=$experiment"
  fi
  while true; do
    heartbeat
    status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep_unlocked 120; continue; }
    printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$fit/status.log"
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
    then break; fi
    observe_finalfit_progress "$experiment" "$status"
    progress_rc=$?
    if (( progress_rc == 2 )); then sleep_unlocked 120; continue; fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
    then sleep_unlocked 120; continue; fi
    if grep -Eqi '\bpaused\b' <<< "$status"; then
      timeout 90 amlt resume "$experiment" ":$job" >> "$fit/controller.log" 2>&1 || true
      sleep_unlocked 120; continue
    fi
    timeout 90 amlt show "$experiment" > "$fit/$experiment.show.txt" 2>&1 || true
    timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
      > "$fit/$experiment.failure.log" 2>&1 || true
    (( wave < max_waves )) || fail "matched final fit exhausted recovery waves"
    wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260810"
    while ! submit_wave "$experiment"; do
      heartbeat; log "waiting to submit matched final-fit recovery wave=$wave"; sleep_unlocked 120
    done
    log "submitted matched final-fit recovery wave=$wave experiment=$experiment"
  done
  rm -rf "$fit/download"; mkdir -p "$fit/download"
  amlt results download "$experiment" --status pass --include '*.json' \
    --output "$fit/download" >> "$fit/controller.log" 2>&1 \
    || fail "matched final-fit result download failed"
  mapfile -t receipts < <(find "$fit/download" -type f -name state8_final_fit_receipt.json)
  mapfile -t publishes < <(find "$fit/download" -type f -name canonical_publish_receipt.json)
  [[ ${#receipts[@]} -eq 1 && ${#publishes[@]} -eq 1 ]] \
    || fail "matched final-fit receipt cardinality drift"
  "$python_bin" - "${receipts[0]}" "${publishes[0]}" "$selection" <<'PY' \
    || fail "matched final-fit receipt validation failed"
import json,sys
r=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); s=json.load(open(sys.argv[3]))['selected']
assert r['stage']=='state8_qualified_unified_final_fit_v1' and r['status']=='final_fit_complete',r
assert r['selected_replay_variant']=='diagnosis_first_matched_seed67',r
assert r['one_epoch_exact'] is True and r['real_examples']==259845,r
assert r['initial_artifact_sha256']==s['artifact_sha256'],(r,s)
assert r['state_shape']==[64,8,5120],r
assert r['history_training_mode']=='strict_state_only' and r['history_dropout_counts']=={'1.0':259845},r
assert r['raw_history_visible_to_decoder'] is False and r['persistent_history_kv_cache'] is False,r
assert float(r['online_kd_weight'])==1.0 and float(r['label_auxiliary_weight'])==1.0,r
assert float(r['trajectory_weight'])==0.1,r
assert r['diagnosis_field_order']==['diagnosis','evidence','label'],r
assert r['canonical_label_role']=='auxiliary_ordinary_weight',r
assert r['causal_output_schema']=='diagnosis_evidence_label_v1',r
assert r['teacher_prompt_provenance_stage']=='state8_diagnosis_first_teacher_full_provenance_gate_v1',r
assert p['stage']=='state8_canonical_final_checkpoint_publish_v1' and p['status']=='complete',p
assert p['selected_replay_variant']==r['selected_replay_variant'],(p,r)
assert p['artifact_sha256']=={'adapter_model':r['adapter_sha256'],'recurrent_state':r['recurrent_state_sha256']},(p,r)
PY
  touch "$fit/scientific_gate_pass" "$fit/complete"
  log "matched exact final fit completed and verified"
fi

if [[ ! -f "$eval_state/complete" ]]; then
  FIT_DIR="$fit" STATE_DIR="$eval_state" FINAL_EVAL_CONFIG="$eval_config" \
    BASE_EXPERIMENT=state8-diagnosis-first-matched-complete-final-eval-20260810 \
    RECOVERY_EXPERIMENT_PREFIX=state8-diagnosis-first-matched-final-eval-recovery \
    RECOVERY_EXPERIMENT_SUFFIX=-20260810 \
    bash scripts/watch_state8_final_eval.sh \
    || fail "matched final evaluation controller crashed"
fi
[[ -f "$eval_state/scientific_gate_pass" ]] || {
  touch "$state/scientific_gate_failed"
  fail "matched exact model failed the complete final gate"
}

while [[ ! -f "$c3/complete" ]]; do
  heartbeat
  [[ ! -f "$c3/needs_attention" ]] || fail "C3 exposure-matched controller needs attention"
  sleep_unlocked 60
done
selection="$upstream/selection.json"
final_gate="$eval_state/final_gate.json"
c3_summary="$c3/summary.json"
mapfile -t publish < <(find "$fit/download" -type f -name canonical_publish_receipt.json)
[[ ${#publish[@]} -eq 1 ]] || fail "matched canonical publish receipt missing"
"$python_bin" scripts/validate_complete_unified_objective.py \
  --selection "$selection" --final-gate "$final_gate" \
  --publish-receipt "${publish[0]}" --c3-summary "$c3_summary" \
  --output "$state/final_objective.json" > "$state/final_objective.log" 2>&1
rc=$?
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"; fail "top-level matched objective gate failed"
else fail "top-level matched objective validator crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "diagnosis-first matched complete unified model objective passed"
