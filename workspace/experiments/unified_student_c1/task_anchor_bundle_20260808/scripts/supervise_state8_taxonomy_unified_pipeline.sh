#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_taxonomy_unified_pipeline_20260807"
poll_seconds="${STATE8_PIPELINE_RESTART_SECONDS:-120}"
mkdir -p "$state"
cd "$project" || exit 1

log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/supervisor.log"; }

while [[ ! -f "$state/complete" ]]; do
  if [[ -f "$state/scientific_gate_failed" ]]; then
    log "scientific gate failed; stopping automatic restart for diagnosis"
    exit 2
  fi
  log "starting unified taxonomy pipeline controller"
  bash scripts/watch_state8_taxonomy_unified_pipeline.sh \
    >> "$state/controller.supervised.log" 2>&1
  rc=$?
  if [[ -f "$state/complete" ]]; then
    break
  fi
  if [[ -f "$state/scientific_gate_failed" ]]; then
    log "controller reached a scientific failure rc=$rc"
    exit 2
  fi
  log "controller infrastructure exit rc=$rc; retrying in ${poll_seconds}s"
  sleep "$poll_seconds"
done

log "unified taxonomy pipeline reached terminal completion"
