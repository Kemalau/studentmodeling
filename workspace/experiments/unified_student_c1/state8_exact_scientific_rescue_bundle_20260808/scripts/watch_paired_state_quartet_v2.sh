#!/usr/bin/env bash
set -uo pipefail

experiment="paired-state-quartet-v2-maskfix-seed42-20260730"
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
monitor_dir="$project_dir/results/paired_state_quartet_v2_maskfix_monitor_20260730"
download_dir="$project_dir/results/paired_state_quartet_v2_maskfix_download_20260730"
status_log="$monitor_dir/status.log"
mkdir -p "$monitor_dir" "$download_dir"
rm -f "$monitor_dir/complete" "$monitor_dir/needs_attention"

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
  echo "$status" > "$monitor_dir/terminal_status.txt"
  if grep -Eqi '3 jobs — 3 pass|3 jobs - 3 pass' <<< "$status"; then
    amlt results download "$experiment" \
      --status pass \
      --include 'gate_summary.json' \
      --include 'run_config.json' \
      --include 'input_manifest.txt' \
      --output "$download_dir" >> "$status_log" 2>&1 || {
        touch "$monitor_dir/needs_attention"; exit 1;
      }
    python scripts/compare_paired_state_quartet.py \
      --input-root "$download_dir" \
      --output-json "$monitor_dir/comparison.json" \
      --output-md "$monitor_dir/comparison.md" >> "$status_log" 2>&1 || {
        touch "$monitor_dir/needs_attention"; exit 1;
      }
    if python - "$monitor_dir/comparison.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("scientific_gate_pass") else 1)
PY
    then
      touch "$monitor_dir/complete"
      exit 0
    fi
    touch "$monitor_dir/scientific_gate_failed"
    exit 2
  fi
  for job in quartet-control-seed42 quartet-swap-all-seed42 quartet-swap-latehalf-seed42; do
    amlt logs view -n 400 "$experiment" ":$job" > "$monitor_dir/$job.log" 2>&1 || true
  done
  touch "$monitor_dir/needs_attention"
  exit 1
done
