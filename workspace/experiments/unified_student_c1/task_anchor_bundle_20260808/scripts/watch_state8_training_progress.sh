#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-keyflip-feeds}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
poll_seconds="${POLL_SECONDS:-300}"
stall_seconds="${STALL_SECONDS:-3600}"
max_run_seconds=172800

declare -A active_files=(
  [primary]="$project/results/state8_qualified_teacher_pipeline_20260805/primary_curriculum/active_experiment.txt"
  [balanced]="$project/results/state8_qualified_balanced_teacher_pipeline_20260805/active_experiment.txt"
  [highkd]="$project/results/state8_qualified_highkd_teacher_pipeline_20260805/active_experiment.txt"
  [repair_trajectory01]="$project/results/state8_stateonly_repair_ab_20260806/active_experiment.txt"
  [repair_logits]="$project/results/state8_stateonly_repair_ab_20260806/active_experiment.txt"
)
declare -A state_dirs=(
  [primary]="$project/results/state8_qualified_teacher_pipeline_20260805/primary_curriculum/progress"
  [balanced]="$project/results/state8_qualified_balanced_teacher_pipeline_20260805/progress"
  [highkd]="$project/results/state8_qualified_highkd_teacher_pipeline_20260805/progress"
  [repair_trajectory01]="$project/results/state8_stateonly_repair_ab_20260806/progress/trajectory01"
  [repair_logits]="$project/results/state8_stateonly_repair_ab_20260806/progress/logits"
)
declare -A repair_jobs=(
  [repair_trajectory01]=state8-repair-trajectory01-seed44
  [repair_logits]=state8-repair-logits-seed44
)

controller="$project/results/state8_training_progress_watchdog_20260805"
mkdir -p "$controller"
cd "$project" || exit 1
exec 9>"$controller/controller.lock"; flock -n 9 || exit 0
echo $$ > "$controller/controller.active.pid"
trap 'rm -f "$controller/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$controller/controller.heartbeat"; }

status_name(){
  "$python_bin" - "$1" <<'PY'
import re,sys
text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',sys.argv[1])
matches=re.findall(r'\b(preparing|queued|running|paused|resuming|pass|passed|failed|killed|cancelled)\b',text,re.I)
print(matches[-1].lower().replace('passed','pass') if matches else 'unknown')
PY
}

update_flags(){
  local state_dir="$1" status="$2"
  if "$python_bin" - "$state_dir/progress.json" "$stall_seconds" "$status" <<'PY'
import json,sys
try: value=json.load(open(sys.argv[1],encoding='utf-8'))
except Exception: raise SystemExit(1)
stalled=(sys.argv[3]=='running' and
         float(value.get('seconds_since_progress') or 0)>=float(sys.argv[2]))
raise SystemExit(0 if stalled else 1)
PY
  then cp "$state_dir/progress.json" "$state_dir/stalled.json"
  else rm -f "$state_dir/stalled.json"; fi
  if "$python_bin" - "$state_dir/progress.json" <<'PY'
import json,sys
try: value=json.load(open(sys.argv[1],encoding='utf-8'))
except Exception: raise SystemExit(1)
raise SystemExit(0 if (value.get('health') or {}).get('healthy') is False else 1)
PY
  then cp "$state_dir/progress.json" "$state_dir/health_alert.json"
  else rm -f "$state_dir/health_alert.json"; fi
}

observe_arm(){
  local arm="$1" active_file="${active_files[$1]}" state_dir="${state_dirs[$1]}"
  local experiment status status_value listing log_file total_rounds=1250 job_args=()
  mkdir -p "$state_dir"
  [[ -s "$active_file" ]] || return 0
  experiment="$(<"$active_file")"
  if [[ "$arm" == repair_* ]]; then
    total_rounds=250
    job_args=(":${repair_jobs[$arm]}")
  fi
  [[ "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || return 0
  status="$(timeout 90 amlt status "$experiment" "${job_args[@]}" 2>&1)" || {
    rm -f "$state_dir/stalled.json"; return 0;
  }
  printf '[%s] arm=%s experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$arm" "$experiment" "$status" >> "$controller/status.log"
  grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' \
    <<< "$status" && { rm -f "$state_dir/stalled.json"; return 0; }
  status_value="$(status_name "$status")"

  if [[ "$status_value" != running ]]; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "" \
      --status "$status_value" --output "$state_dir/progress.json" \
      --total-rounds "$total_rounds" >> "$controller/controller.log" 2>&1 || true
    rm -f "$state_dir/stalled.json" "$state_dir/health_alert.json"
    return 0
  fi

  listing="$(timeout 90 amlt logs list "$experiment" "${job_args[@]}" 2>&1)" || listing=""
  log_file="$("$python_bin" - "$listing" <<'PY'
import re,sys
paths=[]
for line in sys.argv[1].splitlines():
    value=line.strip()
    if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
        match=re.search(r'retry_(\d+)',value)
        paths.append((int(match.group(1)) if match else -1,value))
print(max(paths)[1] if paths else '')
PY
)"
  if [[ -n "$log_file" ]] && timeout 90 amlt logs view -F "$log_file" -n 500 \
      "$experiment" "${job_args[@]}" > "$state_dir/latest_rank0.log" 2> "$state_dir/latest_rank0.stderr"; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --log "$state_dir/latest_rank0.log" --experiment "$experiment" \
      --log-file "$log_file" --status "$status_value" --output "$state_dir/progress.json" \
      --total-rounds "$total_rounds" --max-run-seconds "$max_run_seconds" \
      >> "$controller/controller.log" 2>&1 || true
  else
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "$log_file" \
      --status "$status_value" --output "$state_dir/progress.json" \
      --total-rounds "$total_rounds" >> "$controller/controller.log" 2>&1 || true
  fi
  update_flags "$state_dir" "$status_value"
}

while true; do
  heartbeat
  for arm in primary balanced highkd repair_trajectory01 repair_logits; do
    observe_arm "$arm"
    heartbeat
  done
  sleep "$poll_seconds"
done
