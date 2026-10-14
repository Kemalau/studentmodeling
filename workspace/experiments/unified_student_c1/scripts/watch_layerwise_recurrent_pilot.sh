#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_recurrent_pilot"
experiments=(
  recurrent-layerwise-pilot-preserve-low-lr-seed42-20260731-split3
  recurrent-layerwise-pilot-balanced-seed42-20260731-split3
  recurrent-layerwise-pilot-specificity-seed42-20260731-split3
)
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

while true; do
  heartbeat
  active=0
  passed=0
  : > "$state_dir/split_status.latest"
  for experiment in "${experiments[@]}"; do
    status="$(amlt status "$experiment" 2>&1)"
    rc=$?
    printf '[%s] experiment=%s rc=%s\n%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$rc" "$status" \
      >> "$state_dir/status.log"
    printf 'experiment=%s rc=%s\n%s\n' "$experiment" "$rc" "$status" \
      >> "$state_dir/split_status.latest"
    if (( rc != 0 )); then
      if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
        active=$((active + 1))
        continue
      fi
      fail "could not read split pilot status for $experiment"
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    then
      passed=$((passed + 1))
    elif grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      active=$((active + 1))
    else
      printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
      amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" \
        > "$state_dir/${experiment}.failure.log" 2>&1 || true
      fail "split layerwise pilot did not pass: $experiment"
    fi
  done
  if (( passed == ${#experiments[@]} )); then
    break
  fi
  log "split pilots pending: passed=$passed active=$active total=${#experiments[@]}"
  sleep 60
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download --status pass \
  "${experiments[@]}" \
  --include '*.json' \
  --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "split pilot result download failed"
python scripts/select_layerwise_recurrent_pilot.py \
  --root "$state_dir/download" --output "$state_dir/selection.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || { touch "$state_dir/scientific_gate_failed"; fail "no split pilot passed frozen causal gate"; }
touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
# These downstream attention files only recorded an unmet pilot dependency;
# clear them once the frozen selection gate has actually passed.
rm -f \
  "$project_dir/results/layerwise_unified_final_pipeline_20260731/final_fit/needs_attention" \
  "$project_dir/results/layerwise_unified_final_pipeline_20260731/final_eval/needs_attention" \
  "$project_dir/results/layerwise_unified_final_pipeline_20260731/t3_semantic_judge/needs_attention"
log "split layerwise recurrent pilot selection complete"
