#!/usr/bin/env bash
set -uo pipefail

experiment="state-visionpad-checkpoint-causal-curve-20260729"
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
run_dir="$project_dir/results/state_visionpad_checkpoint_causal_curve_monitor_20260729"
download_dir="$project_dir/results/state_visionpad_checkpoint_causal_curve_download_20260729"
status_log="$run_dir/status.log"
terminal_file="$run_dir/terminal_status.txt"
mkdir -p "$run_dir" "$download_dir"
rm -f "$run_dir/needs_attention" "$run_dir/complete"

cd "$project_dir"
while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status="$(amlt status "$experiment" 2>&1)"
  {
    echo "[$timestamp]"
    echo "$status"
  } >> "$status_log"

  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out' <<< "$status"; then
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
      --include 'curve_summary.json' \
      --include 'curve_summary.md' \
      --include 'run_config.json' \
      --output "$download_dir" >> "$status_log" 2>&1
    python scripts/compare_checkpoint_causal_curves.py \
      --input-root "$download_dir" \
      --output-json "$run_dir/comparison.json" \
      --output-md "$run_dir/comparison.md" >> "$status_log" 2>&1
    touch "$run_dir/complete"
    exit 0
  fi

  for job in original-visionpad-seed42-causal-curve cf1-visionpad-seed42-causal-curve; do
    amlt logs view -n 300 "$experiment" ":$job" > "$run_dir/$job.log" 2>&1 || true
  done
  touch "$run_dir/needs_attention"
  exit 1
done
