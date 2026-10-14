#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_g1_sensitivity"
config="$project_dir/amlt/feeds_m4_causal_bootstrap_g1_sensitivity_20260731.yaml"
job="m4-causal-bootstrap-sum-w1-g1-sensitivity-seed42"
base_experiment="recurrent-m4-causal-bootstrap-g1-sensitivity-seed42-20260731"
mkdir -p "$state_dir/download"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

experiment_class() {
  local experiment="$1" output rc
  output="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  python - "$rc" "$output" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
}

wait_quota() {
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1]))
q = row['quotas_(used/limit)']
assert q['standard']['limit'] - q['standard']['used'] >= 1
assert q['user_max']['limit'] - (q['user_max'].get('used') or 0) >= 1
PY
    then return 0; fi
    sleep 120
  done
}

submit_if_missing() {
  local experiment="$1" class output rc dump
  class="$(experiment_class "$experiment")" || fail "ambiguous experiment state: $experiment"
  [[ "$class" != exists ]] || return 0
  [[ "$class" == missing ]] || fail "unexpected experiment class=$class: $experiment"
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  dump="$state_dir/${experiment}.dump.yaml"
  amlt run "$config" --dump -t Feeds > "$dump" 2>&1 \
    || { flock -u 8; fail "AMLT dump failed: $experiment"; }
  python scripts/validate_m4_causal_bootstrap_g1_sensitivity.py --config "$dump" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "preflight failed: $experiment"; }
  output="$(amlt run "$config" "$experiment" -t Feeds -y \
    --description 'Recovery of isolated one-A100 donor-bootstrap topology sensitivity; resumes latest atomic checkpoint' 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || fail "submission failed: $experiment"
}

active_experiment="$base_experiment"
for recovery_wave in 0 1 2 3; do
  (( recovery_wave == 0 )) \
    || active_experiment="${base_experiment}-recovery${recovery_wave}"
  submit_if_missing "$active_experiment"
  while true; do
    heartbeat
    status="$(amlt status "$active_experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$active_experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
      sleep 120
      continue
    fi
    if grep -Fqi 'No jobs found in the current experiment' <<< "$status"; then
      sleep 30
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    then
      printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
      amlt results download "$active_experiment" --status pass \
        --include 'branch_training_summary.json' \
        --include 'validation_metrics.json' \
        --include 'causal_validation_metrics.json' \
        --include 'training_metrics.json' \
        --include 'run_config.json' \
        --include 'recurrent_causal_gate.json' \
        --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
        || fail "result download failed"
      python scripts/summarize_m4_causal_strength_branches.py \
        --root "$state_dir/download" \
        --expected-branches bootstrap_sum_w1_g1_sensitivity \
        --output-json "$state_dir/comparison.json" \
        --output-md "$state_dir/comparison.md" \
        >> "$state_dir/controller.log" 2>&1 \
        || fail "result summary failed"
      if python - "$state_dir/comparison.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
raise SystemExit(0 if x.get('scientific_gate_pass') is True else 1)
PY
      then
        touch "$state_dir/scientific_gate_passed"
        log "G1 bootstrap sensitivity passed its diagnostic C4 gate"
      else
        touch "$state_dir/scientific_gate_failed"
        log "G1 bootstrap sensitivity completed but missed its diagnostic C4 gate"
      fi
      touch "$state_dir/complete"
      exit 0
    fi
    printf '%s\n' "$status" > "$state_dir/${active_experiment}.terminal_status.txt"
    amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
    timeout 180s amlt logs view -n 1200 "$active_experiment" \
      > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
    break
  done
  (( recovery_wave < 3 )) || fail "G1 sensitivity exhausted recovery waves"
done
