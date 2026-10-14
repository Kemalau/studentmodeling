#!/usr/bin/env bash
set -uo pipefail

experiment="paired-state-swap-ab-seed42-20260730"
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
monitor_dir="$project_dir/results/paired_state_swap_ab_monitor_20260730"
download_dir="$project_dir/results/paired_state_swap_ab_download_20260730"
status_log="$monitor_dir/status.log"
terminal_file="$monitor_dir/terminal_status.txt"
mkdir -p "$monitor_dir" "$download_dir"
rm -f "$monitor_dir/complete" "$monitor_dir/needs_attention"

cd "$project_dir"
while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Refresh the backend status. --no-update-db can remain stale for hours on
  # Singularity and previously reported Running jobs as Preparing.
  status="$(amlt status "$experiment" 2>&1)"
  {
    echo "[$timestamp]"
    echo "$status"
  } >> "$status_log"

  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
    sleep 120
    continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming' <<< "$status"; then
    sleep 120
    continue
  fi

  echo "$status" > "$terminal_file"
  if grep -Eqi '2 jobs — 2 pass|2 jobs - 2 pass' <<< "$status"; then
    amlt results download "$experiment" \
      --status pass \
      --include 'gate_summary.json' \
      --include 'run_config.json' \
      --include 'input_manifest.txt' \
      --output "$download_dir" >> "$status_log" 2>&1
    python scripts/compare_paired_state_swap_ab.py \
      --input-root "$download_dir" \
      --output-json "$monitor_dir/comparison.json" \
      --output-md "$monitor_dir/comparison.md" >> "$status_log" 2>&1
    touch "$monitor_dir/complete"
    exit 0
  fi

  for job in paired-state-control-seed42 paired-state-swap-seed42; do
    amlt logs view -n 300 "$experiment" ":$job" > "$monitor_dir/$job.log" 2>&1 || true
  done
  touch "$monitor_dir/needs_attention"
  exit 1
done
