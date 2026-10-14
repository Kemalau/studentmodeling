#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_strength_escalation"
base_config="$project_dir/amlt/feeds_m4_causal_strength_branches_20260731.yaml"
config="$project_dir/amlt/feeds_m4_causal_strength_escalation_20260731.yaml"
base_experiment="recurrent-m4-causal-strength-escalation-round250-to500-seed42-20260731"
jobs=(m4-causal-raw8-round250-to500-seed42 m4-causal-mean2-normalized-round250-to500-seed42)
mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

wait_quota() {
  local required="$1"
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" "$state_dir/quota_before_submission.json" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1])); q = row['quotas_(used/limit)']
required = int(sys.argv[2])
available = int(q['standard']['limit']) - int(q['standard']['used'])
user_available = int(q['user_max']['limit']) - int(q['user_max'].get('used') or 0)
open(sys.argv[3], 'w', encoding='utf-8').write(json.dumps({
    'selected_row': row, 'required_units': required,
    'standard_available': available, 'user_available': user_available,
}, indent=2, sort_keys=True) + '\n')
assert available >= required and user_available >= required
PY
    then return 0; fi
    sleep 120
  done
}

render_and_validate() {
  python scripts/render_m4_causal_strength_escalation.py \
    --base "$base_config" --output "$config" \
    >> "$state_dir/controller.log" 2>&1 \
    || fail "could not render escalation config"
  python scripts/validate_m4_causal_strength_escalation_amlt_config.py \
    --config "$config" --jobs "${jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || fail "escalation config validation failed"
}

download_passed() {
  local experiment="$1" download="$state_dir/download"
  mkdir -p "$download"
  local status
  status="$(amlt status "$experiment" 2>&1)" || return 1
  if ! grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status"; then
    log "no passed siblings to download from $experiment"
    return 0
  fi
  amlt results download "$experiment" --status pass \
    --include 'branch_training_summary.json' --include 'validation_metrics.json' \
    --include 'causal_validation_metrics.json' --include 'recurrent_causal_gate.json' \
    --output "$download" >> "$state_dir/controller.log" 2>&1
}

submit() {
  local experiment="$1"; shift
  local selected=("$@") selectors=() job
  for job in "${selected[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.dump.yaml" 2>&1 \
    || { flock -u 8; fail "escalation AMLT dump failed"; }
  python scripts/validate_m4_causal_strength_escalation_amlt_config.py \
    --config "$state_dir/${experiment}.dump.yaml" --jobs "${selected[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "escalation dump validation failed"; }
  status="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || { flock -u 8; fail "could not classify escalation experiment"; }
  if [[ "$class" == missing ]]; then
    amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Escalated controlled M4 causal strength after lower doses miss frozen C4; jobs=${selected[*]}" \
      > "$state_dir/${experiment}.submission.txt" 2>&1 \
      || { flock -u 8; fail "escalation submission failed"; }
  elif [[ "$class" != exists ]]; then
    flock -u 8
    fail "ambiguous pre-existing escalation experiment"
  fi
  flock -u 8
}

render_and_validate
wait_quota 8
active_experiment="$base_experiment"
active_jobs=("${jobs[@]}")
submit "$active_experiment" "${active_jobs[@]}"

for recovery_wave in 0 1 2 3; do
  while true; do
    heartbeat
    status="$(amlt status "$active_experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$active_experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
    if python - "$status" "${#active_jobs[@]}" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
    then
      printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
      download_passed "$active_experiment" \
        || fail "could not download passed escalation results from $active_experiment"
      download="$state_dir/download"
      python scripts/summarize_m4_causal_strength_branches.py --root "$download" \
        --expected-branches raw8 mean2_normalized \
        --output-json "$state_dir/comparison.json" --output-md "$state_dir/comparison.md" \
        >> "$state_dir/controller.log" 2>&1 \
        || fail "could not summarize escalation results"
      touch "$state_dir/complete"
      log "causal-strength escalation comparison complete"
      exit 0
    fi
    break
  done
  printf '%s\n' "$status" > "$state_dir/${active_experiment}.terminal_status.txt"
  download_passed "$active_experiment" \
    || fail "could not download passed escalation results from $active_experiment"
  amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
  amlt logs view -n 1200 "$active_experiment" > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]), sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal escalation state has no recoverable jobs"
  (( recovery_wave < 3 )) || fail "escalation exhausted three recovery waves"
  wait_quota "$((4 * ${#failed_jobs[@]}))"
  active_experiment="recurrent-m4-causal-strength-escalation-recovery$((recovery_wave + 1))-seed42-20260731"
  active_jobs=("${failed_jobs[@]}")
  submit "$active_experiment" "${active_jobs[@]}"
done
