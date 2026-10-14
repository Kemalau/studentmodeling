#!/usr/bin/env bash
set -uo pipefail

experiment="recurrent-m2-factorial-seed42-v4-20260730"
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
monitor_dir="$project_dir/results/recurrent_m2_factorial_monitor_20260730"
download_dir="$project_dir/results/recurrent_m2_factorial_download_20260730"
status_log="$monitor_dir/status.log"
mkdir -p "$monitor_dir" "$download_dir"
rm -f "$monitor_dir/complete" "$monitor_dir/needs_attention"

jobs=(
  recurrent-u1-identity-seed42
  recurrent-u1-trainable-seed42
  recurrent-u4-identity-seed42
  recurrent-u4-trainable-seed42
)

cd "$project_dir"
while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status="$(amlt status "$experiment" 2>&1)"
  { echo "[$timestamp]"; echo "$status"; } >> "$status_log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out' <<< "$status"; then
    sleep 120
    continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming' <<< "$status"; then
    sleep 120
    continue
  fi
  echo "$status" > "$monitor_dir/terminal_status.txt"
  if grep -Eqi '4 jobs — 4 pass|4 jobs - 4 pass' <<< "$status"; then
    amlt results download "$experiment" \
      --status pass \
      --include 'm2_gate.json' \
      --include 'validation_metrics.json' \
      --include 'gradient_audit.json' \
      --include 'completion.json' \
      --include 'run_config.json' \
      --include 'data_stats.json' \
      --output "$download_dir" >> "$status_log" 2>&1 || {
        touch "$monitor_dir/needs_attention"; exit 1;
      }
    python scripts/summarize_recurrent_factorial.py \
      --input-root "$download_dir" \
      --output-json "$monitor_dir/summary.json" \
      --output-md "$monitor_dir/summary.md" >> "$status_log" 2>&1
    if python - "$monitor_dir/summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("downstream_rescue_authorized") else 1)
PY
    then
      touch "$monitor_dir/complete"
      exit 0
    fi
    touch "$monitor_dir/scientific_gate_failed"
    exit 2
  fi
  for job in "${jobs[@]}"; do
    amlt logs view -n 400 "$experiment" ":$job" > "$monitor_dir/$job.log" 2>&1 || true
  done
  touch "$monitor_dir/needs_attention"
  exit 1
done
