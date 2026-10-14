#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_strong1_widen8_bridge_20260808/complete_pipeline"
mkdir -p "$state"
cd "$project" || exit 1

while [[ ! -f "$state/complete" ]]; do
  bash scripts/supervise_state8_strong1_widen8_complete.sh \
    >> "$state/supervisor.log" 2>&1
  rc=$?
  [[ -f "$state/complete" ]] && break
  if [[ -f "$state/scientific_gate_failed" ]]; then exit 2; fi
  printf '[%s] recoverable complete-supervisor exit rc=%s; retrying\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >> "$state/supervisor.log"
  sleep 120
done
