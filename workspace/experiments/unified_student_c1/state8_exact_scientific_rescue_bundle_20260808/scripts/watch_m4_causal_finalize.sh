#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_strength_branches"
config="amlt/feeds_m4_causal_strength_finalize_20260731.yaml"
base_experiment="recurrent-m4-causal-strength-finalize-seed42-20260731"
mkdir -p "$state_dir/finalize_download"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
cd "$project_dir"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/finalize_controller.log"; }
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
rm -f "$state_dir/needs_attention"

download_passed() {
  local experiment="$1" status
  status="$(amlt status "$experiment" 2>&1)" || return 1
  grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status" || return 0
  amlt results download "$experiment" --status pass \
    --include 'branch_training_summary.json' \
    --include 'validation_metrics.json' \
    --include 'causal_validation_metrics.json' \
    --include 'recurrent_causal_gate.json' \
    --output "$state_dir/finalize_download" >> "$state_dir/finalize_controller.log" 2>&1
}

wait_quota() {
  local required="$1"
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
q = select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']
required = int(sys.argv[2])
assert q['standard']['limit'] - q['standard']['used'] >= required
assert q['user_max']['limit'] - (q['user_max'].get('used') or 0) >= required
PY
    then return 0; fi
    sleep 120
  done
}

active_experiment="$base_experiment"
expected_jobs=2
for recovery in 0 1 2 3; do
  while true; do
    heartbeat
    status="$(amlt status "$active_experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$active_experiment" "$status" \
      >> "$state_dir/finalize_status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
    if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
    then
      download_passed "$active_experiment" || fail "could not download finalization results"
      python scripts/summarize_m4_causal_strength_branches.py \
        --root "$state_dir/finalize_download" \
        --output-json "$state_dir/comparison.json" \
        --output-md "$state_dir/comparison.md" \
        >> "$state_dir/finalize_controller.log" 2>&1 \
        || fail "could not summarize finalization results"
      touch "$state_dir/complete"
      log "single-GPU causal-strength finalization complete"
      exit 0
    fi
    break
  done
  download_passed "$active_experiment" || fail "could not preserve passed finalization sibling"
  amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
  amlt logs view -n 1200 "$active_experiment" > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]), sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal finalization state has no recoverable jobs"
  (( recovery < 3 )) || fail "single-GPU finalization exhausted recovery waves"
  wait_quota "${#failed_jobs[@]}"
  active_experiment="recurrent-m4-causal-strength-finalize-recovery$((recovery + 1))-seed42-20260731"
  selectors=(); for job in "${failed_jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$state_dir/${active_experiment}.dump.yaml" \
    || { flock -u 8; fail "finalization recovery dump failed"; }
  python scripts/validate_m4_causal_finalize_amlt_config.py \
    --config "$state_dir/${active_experiment}.dump.yaml" --jobs "${failed_jobs[@]}" \
    >> "$state_dir/finalize_controller.log" 2>&1 \
    || { flock -u 8; fail "finalization recovery preflight failed"; }
  amlt run "$config" "${selectors[@]}" "$active_experiment" -t Feeds -y \
    --description "Single-GPU round-500 checkpoint finalization recovery=$((recovery + 1))" \
    > "$state_dir/${active_experiment}.submission.txt" 2>&1 \
    || { flock -u 8; fail "finalization recovery submission failed"; }
  flock -u 8
  expected_jobs="${#failed_jobs[@]}"
done
