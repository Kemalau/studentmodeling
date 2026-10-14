#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=qwen-loo-t1-t5-1gpu-seed42-20260826
t5_recovery=qwen-loo-t5-1gpu-recovery1-seed42-20260826
state="$project/results/qwen_loo_seed42_1gpu_20260826"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds-v2
mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT

while true; do
  date -u +%FT%TZ > "$state/watcher.heartbeat"
  amlt --json-tables list "$experiment" > "$state/status.main.json.tmp" 2> "$state/status.stderr"
  rc=$?
  if (( rc != 0 )); then
    sleep 60
    continue
  fi
  amlt --json-tables list "$t5_recovery" > "$state/status.t5_recovery.json.tmp" 2>> "$state/status.stderr"
  rc=$?
  if (( rc != 0 )); then
    sleep 60
    continue
  fi
  mv "$state/status.main.json.tmp" "$state/status.main.json"
  mv "$state/status.t5_recovery.json.tmp" "$state/status.t5_recovery.json"
  decision="$(python - "$state/status.main.json" "$state/status.t5_recovery.json" <<'PY'
import json, sys
main = json.load(open(sys.argv[1]))
recovery = json.load(open(sys.argv[2]))
main_status = {
    str(row.get("job_name", "")): str(row.get("status", "")).lower()
    for row in main
}
statuses = [
    status for name, status in main_status.items()
    if "loo-t5-" not in name
]
statuses.extend(str(row.get("status", "")).lower() for row in recovery)
if len(statuses) != 5:
    print("wait")
elif any(status in {"failed", "killed", "canceled", "cancelled"} for status in statuses):
    print("fail")
elif all(status in {"pass", "passed"} for status in statuses):
    print("pass")
else:
    print("wait")
PY
)"
  case "$decision" in
    fail)
      amlt show "$experiment" > "$state/failure.show.txt" 2>&1
      amlt show "$t5_recovery" >> "$state/failure.show.txt" 2>&1
      amlt logs download "$experiment" --output "$state/failure_logs/main" > "$state/failure_logs.download.txt" 2>&1 || true
      amlt logs download "$t5_recovery" --output "$state/failure_logs/t5_recovery" >> "$state/failure_logs.download.txt" 2>&1 || true
      touch "$state/needs_attention"
      exit 1
      ;;
    pass)
      amlt results download "$experiment" --output "$state/download" > "$state/download.log" 2>&1 || {
        sleep 60
        continue
      }
      amlt results download "$t5_recovery" --output "$state/download" >> "$state/download.log" 2>&1 || {
        sleep 60
        continue
      }
      touch "$state/complete"
      exit 0
      ;;
    *) sleep 60 ;;
  esac
done
