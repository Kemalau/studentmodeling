#!/usr/bin/env bash
set -uo pipefail

experiment="recurrent-finalfit-cache-generation-kd-trainable-seed42-v1-20260730"
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_final_pipeline_20260730/speculative_cache_monitor"
status_log="$state_dir/status.log"
jobs=(
  finalfit-cache-rank0-seed42
  finalfit-cache-rank1-seed42
  finalfit-cache-rank2-seed42
  finalfit-cache-rank3-seed42
)

mkdir -p "$state_dir"
rm -f "$state_dir/complete" "$state_dir/needs_attention"
cd "$project_dir"

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status="$(amlt status "$experiment" 2>&1)"
  { echo "[$timestamp]"; echo "$status"; } >> "$status_log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
    sleep 120
    continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming' <<< "$status"; then
    sleep 120
    continue
  fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  if grep -Eqi '4 jobs (—|-)[[:space:]]*4 pass' <<< "$status"; then
    touch "$state_dir/complete"
    exit 0
  fi
  for job in "${jobs[@]}"; do
    amlt logs view -n 800 "$experiment" ":$job" \
      > "$state_dir/$job.log" 2>&1 || true
  done
  touch "$state_dir/needs_attention"
  exit 1
done
