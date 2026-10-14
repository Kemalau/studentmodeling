#!/usr/bin/env bash
set -uo pipefail

primary_experiment="state-alias-factorial-7job-20260729"
retry_experiment="state-alias-factorial-original-retry1-20260729"
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
run_dir="$project_dir/results/state_alias_factorial_monitor_20260729"
download_dir="$project_dir/results/state_alias_factorial_download_20260729"
status_log="$run_dir/status.log"
terminal_file="$run_dir/terminal_status.txt"
mkdir -p "$run_dir" "$download_dir"

cd "$project_dir"
while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  primary_status="$(amlt status "$primary_experiment" 2>&1)"
  retry_status="$(amlt status "$retry_experiment" 2>&1)"
  {
    echo "[$timestamp]"
    echo "=== $primary_experiment ==="
    echo "$primary_status"
    echo "=== $retry_experiment ==="
    echo "$retry_status"
  } >> "$status_log"

  if grep -Eqi 'running|queued|preparing|paused|resuming' <<< "$primary_status$retry_status"; then
    sleep 120
    continue
  fi

  {
    echo "=== $primary_experiment ==="
    echo "$primary_status"
    echo "=== $retry_experiment ==="
    echo "$retry_status"
  } > "$terminal_file"
  if grep -Eqi '7 jobs — 4 pass, 3 failed|7 jobs - 4 pass, 3 failed' <<< "$primary_status" \
      && grep -Eqi '3 jobs — 3 pass|3 jobs - 3 pass' <<< "$retry_status"; then
    amlt results download "$primary_experiment" \
      --status pass \
      --include 'gate/gate_summary.json' \
      --output "$download_dir" >> "$status_log" 2>&1
    amlt results download "$retry_experiment" \
      --status pass \
      --include 'gate/gate_summary.json' \
      --output "$download_dir" >> "$status_log" 2>&1
    python scripts/summarize_state_alias_factorial.py \
      --input-root "$download_dir" \
      --output-json "$run_dir/summary.json" \
      --output-md "$run_dir/summary.md" >> "$status_log" 2>&1
    touch "$run_dir/complete"
    exit 0
  fi

  for job in \
    state-cf1-visionpad-seed13 \
    state-cf1-visionpad-seed97 \
    state-original-visionpad-seed13 \
    state-original-visionpad-seed42 \
    state-original-visionpad-seed97 \
    state-cf1-imagepad-seed42 \
    state-cf1-videopad-seed42; do
    amlt logs view -n 240 "$primary_experiment" ":$job" > "$run_dir/primary-$job.log" 2>&1 || true
    amlt logs view -n 240 "$retry_experiment" ":$job" > "$run_dir/retry-$job.log" 2>&1 || true
  done
  touch "$run_dir/needs_attention"
  exit 1
done
