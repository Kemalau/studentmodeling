#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
checkpoint_round="${CHECKPOINT_ROUND:-1000}"
[[ "$checkpoint_round" == 1000 ]] || exit 2
snapshot_owner="$pipeline/checkpoint${checkpoint_round}_c4"
fit_dir="$pipeline/final_fit"
auth_required="$project_dir/results/recurrent_unified_supervisor_20260731/auth_required"
state_dir="$pipeline/checkpoint${checkpoint_round}_formal_screen"
config="$project_dir/amlt/feeds_layerwise_finalfit_checkpoint250_formal_screen_20260801.yaml"
base_experiment="layerwise-finalfit-checkpoint${checkpoint_round}-formal-screen-seed42-20260801"
active_file="$state_dir/active_experiment.txt"
expected_jobs_file="$state_dir/expected_jobs.txt"
results_root="$state_dir/preserved_results"
if [[ -s "$active_file" ]]; then active_experiment="$(<"$active_file")"; else active_experiment="$base_experiment"; fi
[[ "$active_experiment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
printf '%s\n' "$active_experiment" > "$active_file"
expected_jobs=4
recovery_wave=0
if [[ "$active_experiment" =~ recovery([0-9]+)-seed42 ]]; then
  recovery_wave="${BASH_REMATCH[1]}"
  if [[ -s "$expected_jobs_file" ]]; then expected_jobs="$(<"$expected_jobs_file")"; fi
fi
[[ "$expected_jobs" =~ ^[1-4]$ ]] || exit 2
printf '%s\n' "$expected_jobs" > "$expected_jobs_file"
mkdir -p "$state_dir" "$results_root"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
attention_is_current(){
  local owner="$1" marker="$1/needs_attention" active_pid="$1/controller.active.pid"
  [[ -f "$marker" ]] || return 1
  if [[ -s "$active_pid" && "$active_pid" -nt "$marker" ]]; then
    local pid
    pid="$(<"$active_pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
  fi
  if [[ -s "$owner/active_experiment.txt" \
        && "$owner/active_experiment.txt" -nt "$marker" ]]; then
    return 1
  fi
  return 0
}

wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" "$state_dir/quota_before_submission.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); n=int(sys.argv[2]); q=row['quotas_(used/limit)']
available=int(q['standard']['limit'])-int(q['standard']['used'])
user_available=int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)
open(sys.argv[3],'w',encoding='utf-8').write(json.dumps({'selected_row':row,'required_units':n,'standard_available':available,'user_available':user_available},indent=2,sort_keys=True)+'\n')
assert available>=n and user_available>=n,(available,user_available,n)
PY
    then return 0; fi
    sleep 120
  done
}

preserve_passed(){
  local experiment="$1"
  amlt results download "$experiment" --status pass \
    --include metrics.json --include predictions.jsonl \
    --include predictions_redacted.jsonl \
    --include manifest_audit.json --include scope.txt \
    --output "$results_root" >> "$state_dir/controller.log" 2>&1
}

# The C4 owner first copies the ephemeral training checkpoint into an immutable
# snapshot and validates its round/world-size/size receipts.  Reuse that exact
# snapshot so the causal and performance diagnostics remain matched.
while [[ ! -f "$snapshot_owner/snapshot_complete" ]]; do
  heartbeat
  if [[ -f "$auth_required" ]]; then sleep 60; continue; fi
  attention_is_current "$snapshot_owner" \
    && fail "round-${checkpoint_round} snapshot owner needs attention"
  attention_is_current "$fit_dir" \
    && fail "formal fit failed before round-${checkpoint_round} snapshot"
  sleep 60
done
test -f "$snapshot_owner/snapshot_checkpoint_completion.json" \
  || fail "round-${checkpoint_round} snapshot receipt is missing"
python - "$snapshot_owner/snapshot_checkpoint_completion.json" "$checkpoint_round" <<'PY' \
  || fail "round-${checkpoint_round} snapshot receipt is invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x.get('status')=='complete' and x.get('round')==int(sys.argv[2]),x
assert x.get('world_size')==16,x
PY

while ! az account get-access-token \
    --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
    --query expiresOn -o tsv >/dev/null 2>&1; do
  heartbeat
  sleep 60
done
az account set --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 >/dev/null \
  || fail "fixed subscription unavailable after authentication"

if [[ "$active_experiment" == "$base_experiment" && ! -f "$state_dir/submission.txt" ]]; then
  wait_quota 4
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds > "$state_dir/preflight.dump.yaml" \
    2> "$state_dir/preflight.stderr" \
    || { flock -u 8; fail "checkpoint formal screen dump failed"; }
  python scripts/validate_layerwise_checkpoint250_formal_screen_amlt.py \
    --config "$state_dir/preflight.dump.yaml" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "checkpoint formal screen preflight failed"; }
  current="$(amlt status --no-update-db "$active_experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "checkpoint formal screen status ambiguous"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$active_experiment" -t Feeds -y \
      --description "Diagnostic-only five-task screen of immutable layerwise final-fit round-${checkpoint_round} snapshot" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/submission.txt"
    ((rc==0)) || { flock -u 8; fail "checkpoint formal screen submission failed"; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" > "$state_dir/submission.txt"
  else
    flock -u 8; fail "checkpoint formal screen pre-existing state ambiguous"
  fi
  flock -u 8
fi

while true; do
  heartbeat
  status="$(amlt status "$active_experiment" 2>&1)"
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$active_experiment" "$status" \
    >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
  if grep -Fqi 'No jobs found in the current experiment' <<< "$status"; then sleep 30; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$active_experiment" || fail "could not download passed checkpoint screen results"
    break
  fi
  preserve_passed "$active_experiment" || fail "could not preserve passed checkpoint screen siblings"
  printf '%s\n' "$status" > "$state_dir/${active_experiment}.terminal_status.txt"
  amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$active_experiment" > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal checkpoint screen has no recoverable jobs"
  (( recovery_wave < 3 )) || fail "checkpoint screen exhausted three recovery waves"
  recovery_wave=$((recovery_wave+1)); expected_jobs="${#failed_jobs[@]}"
  printf '%s\n' "$expected_jobs" > "$expected_jobs_file"
  wait_quota "$expected_jobs"
  next_experiment="layerwise-finalfit-checkpoint${checkpoint_round}-formal-screen-recovery${recovery_wave}-seed42-20260801"
  selectors=(); for job in "${failed_jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${next_experiment}.preflight.dump.yaml" \
    2> "$state_dir/${next_experiment}.preflight.stderr" \
    || { flock -u 8; fail "checkpoint screen recovery dump failed"; }
  python scripts/validate_layerwise_checkpoint250_formal_screen_amlt.py \
    --config "$state_dir/${next_experiment}.preflight.dump.yaml" --jobs "${failed_jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "checkpoint screen recovery preflight failed"; }
  current="$(amlt status --no-update-db "$next_experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "checkpoint screen recovery status ambiguous"; }
  if [[ "$class" == missing ]]; then
    amlt run "$config" "${selectors[@]}" "$next_experiment" -t Feeds -y \
      --description "Round-${checkpoint_round} five-task diagnostic recovery=${recovery_wave}; jobs=${failed_jobs[*]}" \
      > "$state_dir/${next_experiment}.submission.txt" 2>&1 \
      || { flock -u 8; fail "checkpoint screen recovery submission failed"; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; fail "checkpoint screen recovery pre-existing state ambiguous"
  fi
  flock -u 8
  active_experiment="$next_experiment"; printf '%s\n' "$active_experiment" > "$active_file"
  log "monitoring checkpoint screen recovery=$recovery_wave jobs=${failed_jobs[*]}"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$results_root/." "$state_dir/download/"
if python scripts/summarize_layerwise_checkpoint250_formal_screen.py \
  --root "$state_dir/download" --output "$state_dir/summary.json" \
  >> "$state_dir/controller.log" 2>&1; then
  touch "$state_dir/diagnostic_gate_pass"
else
  touch "$state_dir/diagnostic_gate_failed"
fi
touch "$state_dir/complete"
log "layerwise round-${checkpoint_round} five-task diagnostic complete"
