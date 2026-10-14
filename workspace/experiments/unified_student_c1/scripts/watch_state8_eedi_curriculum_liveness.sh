#!/usr/bin/env bash
set -uo pipefail

# Scientific liveness guard for the two 5,000-step State8 curricula.
#
# The owning controllers already recover terminal AMLT attempts from atomic
# checkpoints.  This guard supplies the missing transition for a backend job
# that remains labelled Running but stops producing optimizer steps.  Log or
# authentication transport failures are never treated as evidence of a stall.

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
state="$project/results/state8_eedi_curriculum_liveness_20260812"
poll_seconds="${STATE8_EEDI_LIVENESS_POLL_SECONDS:-120}"
stall_seconds="${STATE8_EEDI_LIVENESS_STALL_SECONDS:-2700}"
startup_stall_seconds="${STATE8_EEDI_LIVENESS_STARTUP_STALL_SECONDS:-7200}"
posttrain_stall_seconds="${STATE8_EEDI_LIVENESS_POSTTRAIN_STALL_SECONDS:-21600}"
total_steps=5000

declare -A roots=(
  [generic]="$project/results/state8_diagnosis_first_eedi_curriculum_20260812"
  [current_query]="$project/results/state8_diagnosis_first_eedi_current_query_curriculum_20260812"
)
declare -A jobs=(
  [generic]=state8-diagnosis-first-eedi-curriculum-seed83
  [current_query]=state8-diagnosis-first-eedi-current-query-curriculum-seed83
)

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }

status_phase(){
  STATUS_JSON="$1" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS_JSON'])
states={str(row.get('status') or '').casefold() for row in rows}
if len(rows)==1 and states=={'running'}: print('running')
elif states & {'preparing','pre-queued','queued','resuming','paused'}: print('forward')
elif len(rows)==1 and states=={'pass'}: print('pass')
else: print('terminal')
PY
}

latest_rank0_log(){
  LISTING="$1" "$python_bin" - <<'PY'
import os,re
paths=[]
for line in os.environ['LISTING'].splitlines():
    value=line.strip()
    if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
        retry=re.search(r'retry_(\d+)',value)
        paths.append((int(retry.group(1)) if retry else -1,value))
print(max(paths)[1] if paths else '')
PY
}

observe(){
  local arm="$1" root="${roots[$1]}" job="${jobs[$1]}"
  local arm_state="$state/$1" experiment status phase listing log_file threshold marker
  mkdir -p "$arm_state"
  [[ -s "$root/active_experiment.txt" ]] || return 0
  experiment="$(<"$root/active_experiment.txt")"
  [[ "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || return 0

  status="$(timeout 90 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update 2>/dev/null)" || return 0
  [[ -n "$status" && "$status" != '[]' ]] || return 0
  printf '%s\n' "$status" > "$arm_state/latest_status.json"
  phase="$(status_phase "$status")" || return 0
  if [[ "$phase" != running ]]; then
    rm -f "$arm_state/stalled.json" "$arm_state/health_alert.json"
    return 0
  fi

  listing="$(timeout 90 amlt logs list "$experiment" ":$job" 2>/dev/null)" \
    || return 0
  log_file="$(latest_rank0_log "$listing")" || return 0
  [[ -n "$log_file" ]] || return 0
  timeout 90 amlt logs view -F "$log_file" -n 700 \
    "$experiment" ":$job" > "$arm_state/latest_rank0.log" \
    2> "$arm_state/latest_rank0.stderr" || return 0
  "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
    --log "$arm_state/latest_rank0.log" --experiment "$experiment" \
    --log-file "$log_file" --status running --output "$arm_state/progress.json" \
    --total-rounds "$total_steps" --max-run-seconds 86400 \
    --validation-reserve-seconds 21600 >> "$state/controller.log" 2>&1 \
    || return 0

  if ! "$python_bin" scripts/state8_eedi_curriculum_liveness.py \
      "$arm_state/progress.json" "$stall_seconds" \
      "$startup_stall_seconds" "$posttrain_stall_seconds"
  then
    rm -f "$arm_state/stalled.json" "$arm_state/health_alert.json"
    return 0
  fi

  cp "$arm_state/progress.json" "$arm_state/stalled.json"
  marker="$arm_state/${experiment}.cancel_requested"
  [[ ! -f "$marker" ]] || return 0
  timeout 90 amlt show "$experiment" > "$arm_state/${experiment}.stalled.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
    > "$arm_state/${experiment}.stalled.log" 2>&1 || true
  if timeout 90 amlt cancel "$experiment" ":$job" -y \
      > "$arm_state/${experiment}.stalled.cancel.txt" 2>&1; then
    touch "$marker"
    log "cancelled stalled/unhealthy attempt arm=$arm experiment=$experiment; exact-resume owner will recover"
  else
    log "WARNING: cancellation request failed arm=$arm experiment=$experiment; will retry after next observation"
  fi
}

while :; do
  heartbeat
  observe generic
  heartbeat
  observe current_query
  heartbeat
  if [[ -f "${roots[generic]}/complete" && -f "${roots[current_query]}/complete" ]]; then
    touch "$state/complete"
    log "both curriculum owners reached complete; liveness guard exiting"
    exit 0
  fi
  sleep_unlocked "$poll_seconds"
done
