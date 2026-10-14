#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
fit_dir="${FIT_DIR:-$pipeline/final_fit}"
state_dir="${STATE_DIR:-$pipeline/progress}"
active_file="$fit_dir/active_experiment.txt"
poll_seconds="${POLL_SECONDS:-300}"
stall_seconds="${STALL_SECONDS:-3600}"
total_rounds="${TOTAL_ROUNDS:-16241}"
max_run_seconds="${MAX_RUN_SECONDS:-345600}"
validation_reserve_seconds="${VALIDATION_RESERVE_SECONDS:-21600}"
mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  if [[ ! -s "$active_file" ]]; then sleep "$poll_seconds"; continue; fi
  experiment="$(<"$active_file")"
  if [[ ! "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then sleep "$poll_seconds"; continue; fi
  if ! status="$(timeout 90 amlt status "$experiment" 2>&1)"; then
    # Monitoring transport failure is not evidence that training stalled.
    # Preserve the last real receipt and retry on the next bounded poll.
    rm -f "$state_dir/stalled.json"
    sleep "$poll_seconds"
    continue
  fi
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
    # A stale pre-authentication stall receipt must never survive into a later
    # healthy status poll and race the final-fit recovery owner.
    rm -f "$state_dir/stalled.json"
    sleep "$poll_seconds"
    continue
  fi
  status_name="$(python - "$status" <<'PY'
import re,sys
text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',sys.argv[1])
m=re.search(r'^:\S+\s+.*?\b(preparing|queued|running|paused|resuming|pass|failed|killed|cancelled)\b',text,re.I|re.M)
print(m.group(1).lower() if m else 'unknown')
PY
)"
  # A newly queued/preparing experiment has no rank log yet.  Publish an
  # explicit receipt immediately so the previous failed experiment's round,
  # losses, and ETA cannot remain in the authoritative progress file.
  readarray -t progress_state < <(python - "$state_dir/progress.json" <<'PY'
import json,sys
try: value=json.load(open(sys.argv[1],encoding='utf-8'))
except Exception: value={}
print(value.get('experiment') or '')
print('none' if value.get('round') is None else 'structured')
PY
)
  progress_experiment="${progress_state[0]:-}"
  progress_round_state="${progress_state[1]:-none}"
  if [[ "$progress_experiment" != "$experiment" || "$progress_round_state" == none ]]; then
    python scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "" \
      --status "$status_name" --output "$state_dir/progress.json" \
      --total-rounds "$total_rounds" >> "$state_dir/controller.log" 2>&1 || true
    rm -f "$state_dir/stalled.json" "$state_dir/health_alert.json"
  fi
  if ! listing="$(timeout 90 amlt logs list "$experiment" 2>&1)"; then
    rm -f "$state_dir/stalled.json"
    sleep "$poll_seconds"
    continue
  fi
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$listing"; then
    rm -f "$state_dir/stalled.json"
    sleep "$poll_seconds"
    continue
  fi
  log_file="$(python - "$listing" <<'PY'
import re,sys
paths=[]
for line in sys.argv[1].splitlines():
    value=line.strip()
    if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_00\.txt',value):
        m=re.search(r'retry_(\d+)',value)
        paths.append((int(m.group(1)) if m else -1,value))
print(max(paths)[1] if paths else '')
PY
)"
  if [[ -n "$log_file" ]]; then
    if timeout 90 amlt logs view -F "$log_file" -n 250 "$experiment" > "$state_dir/latest_rank0.log" 2> "$state_dir/latest_rank0.stderr"; then
      python scripts/monitor_layerwise_unified_training_progress.py \
        --log "$state_dir/latest_rank0.log" --experiment "$experiment" \
        --log-file "$log_file" --status "$status_name" --output "$state_dir/progress.json" \
        --total-rounds "$total_rounds" --max-run-seconds "$max_run_seconds" \
        --validation-reserve-seconds "$validation_reserve_seconds" \
        >> "$state_dir/controller.log" 2>&1 || true
      if python - "$state_dir/progress.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
raise SystemExit(0 if (x.get('health') or {}).get('healthy') is False else 1)
PY
      then
        cp "$state_dir/progress.json" "$state_dir/health_alert.json"
      else
        rm -f "$state_dir/health_alert.json"
      fi
      if python - "$state_dir/progress.json" "$stall_seconds" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
raise SystemExit(0 if x.get('amlt_status')=='running' and float(x.get('seconds_since_progress',0))>=float(sys.argv[2]) else 1)
PY
      then
        cp "$state_dir/progress.json" "$state_dir/stalled.json"
      else
        rm -f "$state_dir/stalled.json"
      fi
    fi
  fi
  sleep "$poll_seconds"
done
