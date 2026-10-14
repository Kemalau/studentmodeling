#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/layerwise_trajectory_curriculum_20260731"
experiment="recurrent-layerwise-trajectory-curriculum-seed42-20260731"
recovery_experiment="recurrent-layerwise-trajectory-r64-validation-only-seed42-20260801"
downstream="$project_dir/results/layerwise_unified_final_pipeline_20260731"
mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/scientific_gate_pass" \
  "$state_dir/scientific_gate_failed" "$state_dir/needs_attention"

heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

original_ready=false
while [[ "$original_ready" != true ]]; do
  heartbeat
  status="$(amlt status "$experiment" 2>&1)"; rc=$?
  printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$status" \
    >> "$state_dir/status.log"
  if (( rc != 0 )); then
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
      sleep 120; continue
    fi
    fail "could not read trajectory curriculum status"
  fi
  if python - "$status" <<'PY'
import sys
import re
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", sys.argv[1])
rows = {}
for line in text.splitlines():
    match = re.search(
        r":(trajectory-r256-(?:scale|cosine)-seed42)\s+.*?\b(pass|running|queued|preparing|paused|resuming|failed|killed|cancelled)\b",
        line,
        re.I,
    )
    if match:
        rows[match.group(1)] = match.group(2).lower()
raise SystemExit(0 if len(rows) == 2 and all(value == "pass" for value in rows.values()) else 1)
PY
  then original_ready=true; break; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60; continue
  fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  amlt show --status "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" > "$state_dir/failure.log" 2>&1 || true
  fail "trajectory curriculum did not pass"
done

while true; do
  heartbeat
  status="$(amlt status "$recovery_experiment" 2>&1)"; rc=$?
  printf '[%s] recovery rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$status" \
    >> "$state_dir/status.log"
  if (( rc != 0 )); then
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
      sleep 120; continue
    fi
    fail "could not read r64 recovery status"
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60; continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
  then break; fi
  printf '%s\n' "$status" > "$state_dir/recovery_terminal_status.txt"
  amlt show --status "$recovery_experiment" > "$state_dir/recovery_show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$recovery_experiment" > "$state_dir/recovery_failure.log" 2>&1 || true
  fail "r64 trajectory recovery did not pass"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" \
  :trajectory-r256-scale-seed42 :trajectory-r256-cosine-seed42 \
  --include '*.json' --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 || fail "r256 trajectory result download failed"
amlt results download "$recovery_experiment" --status pass \
  --include '*.json' --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 || fail "r64 trajectory result download failed"
python scripts/select_layerwise_trajectory_curriculum.py \
  --root "$state_dir/download" --output "$state_dir/selection.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || { touch "$state_dir/scientific_gate_failed"; fail "no trajectory branch passed frozen gates"; }
touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
rm -f \
  "$downstream/final_fit/needs_attention" \
  "$downstream/final_eval/needs_attention" \
  "$downstream/t3_semantic_judge/needs_attention"
log "trajectory curriculum selection complete"
