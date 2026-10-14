#!/usr/bin/env bash
set -uo pipefail

projection="${1:?projection is required}"
experiment="${2:?experiment is required}"
if [[ "$projection" != identity && "$projection" != trainable ]]; then
  echo "invalid projection=$projection" >&2
  exit 2
fi
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
monitor_dir="$project_dir/results/recurrent_m3_monitor_20260730"
download_dir="$project_dir/results/recurrent_m3_download_20260730"
status_log="$monitor_dir/status.log"
mkdir -p "$monitor_dir" "$download_dir"
rm -f "$monitor_dir/complete" "$monitor_dir/needs_attention" \
  "$monitor_dir/scientific_gate_failed"
jobs=(
  "m3-generation-only-${projection}-seed42"
  "m3-generation-kd-${projection}-seed42"
)

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
  if grep -Eqi '2 jobs — 2 pass|2 jobs - 2 pass' <<< "$status"; then
    amlt results download "$experiment" \
      --status pass \
      --include 'validation_metrics.json' \
      --include 'gradient_audit.json' \
      --include 'completion.json' \
      --include 'run_config.json' \
      --include 'training_metrics_by_task_source.json' \
      --include 'data_audit.json' \
      --output "$download_dir" >> "$status_log" 2>&1 || {
        touch "$monitor_dir/needs_attention"; exit 1;
      }
    python scripts/summarize_m3_gate.py \
      --input-root "$download_dir" \
      --output-json "$monitor_dir/summary.json" \
      --output-md "$monitor_dir/summary.md" \
      --projection "$projection" >> "$status_log" 2>&1 || {
        touch "$monitor_dir/needs_attention"; exit 1;
      }
    if python - "$monitor_dir/summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
strict = value.get("scientific_gate_pass") is True
rescue = value.get("downstream_rescue_authorized") is True
raise SystemExit(0 if strict or rescue else 1)
PY
    then
      touch "$monitor_dir/complete"
      exit 0
    fi
    touch "$monitor_dir/scientific_gate_failed"
    exit 2
  fi
  for job in "${jobs[@]}"; do
    amlt logs view -n 500 "$experiment" ":$job" > "$monitor_dir/$job.log" 2>&1 || true
  done
  # A user/API cancellation is operational rather than a scientific or code
  # failure. Record it separately so the controller can be restarted safely
  # under a new retry name and reuse the same remote atomic checkpoints.
  if grep -Eqi 'killed|cancelled|canceled' <<< "$status" \
      && ! grep -Eqi 'failed' <<< "$status"; then
    printf '%s\n' "$experiment" > "$monitor_dir/retry_required"
    rm -f "$project_dir/results/recurrent_m3_submit_20260730/controller.active.pid"
    exec 9>&- 2>/dev/null || true
    nohup bash -lc "sleep 5; cd '$project_dir'; bash scripts/start_recurrent_m3_controller.sh" \
      >> "$project_dir/results/recurrent_m3_submit_20260730/auto_retry.log" 2>&1 &
    exit 75
  fi
  touch "$monitor_dir/needs_attention"
  exit 1
done
