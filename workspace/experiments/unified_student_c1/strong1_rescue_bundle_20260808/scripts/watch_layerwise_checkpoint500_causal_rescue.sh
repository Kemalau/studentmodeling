#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/checkpoint500_causal_rescue"
experiment="layerwise-c500-causal-rescue-seed42-20260801"
mkdir -p "$state_dir"; cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(amlt status "$experiment" 2>&1)"
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],2) else 1)
PY
  then break; fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" > "$state_dir/failure.log" 2>&1 || true
  fail "causal-rescue experiment ended without two passed jobs"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include 'recurrent_causal_gate.json' --include 'rescue_training_receipt.json' \
  --include 'completion.json' --include 'run_config.json' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "causal-rescue result download failed"
python scripts/summarize_layerwise_checkpoint500_causal_rescue.py \
  --root "$state_dir/download" --output "$state_dir/comparison.json" \
  >> "$state_dir/controller.log" 2>&1 || fail "causal-rescue summary validation failed"
touch "$state_dir/complete"
log "checkpoint-500 causal-rescue comparison complete"
