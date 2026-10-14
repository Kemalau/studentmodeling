#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_strength_branches"
config="amlt/feeds_m4_causal_strength_branches_20260731.yaml"
base_experiment="recurrent-m4-causal-strength-round250-to500-seed42-20260731"
mkdir -p "$state_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
cd "$project_dir"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
rm -f "$state_dir/needs_attention"

wait_quota() {
  local required="$1"
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1]))
q = row['quotas_(used/limit)']; required = int(sys.argv[2])
assert q['standard']['limit'] - q['standard']['used'] >= required
assert q['user_max']['limit'] - (q['user_max'].get('used') or 0) >= required
PY
    then return 0; fi
    sleep 120
  done
}

download_passed() {
  local experiment="$1" download="$state_dir/download"
  mkdir -p "$download"
  local status
  status="$(amlt status "$experiment" 2>&1)" || return 1
  # A terminal wave can contain no passed jobs even though each job saved a
  # valid atomic checkpoint before rank-zero validation timed out. In that
  # case there is nothing to preserve yet; recover every failed job instead of
  # treating AMLT's "No matching jobs" response as a controller failure.
  if ! grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status"; then
    log "no passed siblings to download from $experiment"
    return 0
  fi
  amlt results download "$experiment" --status pass \
    --include 'branch_training_summary.json' \
    --include 'validation_metrics.json' \
    --include 'causal_validation_metrics.json' \
    --include 'recurrent_causal_gate.json' \
    --output "$download" >> "$state_dir/controller.log" 2>&1
}

active_experiment="$base_experiment"
expected_jobs=2
for recovery_wave in 0 1 2 3; do
  while true; do
    heartbeat
    status="$(amlt status "$active_experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$active_experiment" "$status" \
      >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
    if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
    then
      printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
      download_passed "$active_experiment" \
        || fail "Could not download passed causal-strength results from $active_experiment"
      download="$state_dir/download"
      python scripts/summarize_m4_causal_strength_branches.py \
        --root "$download" \
        --output-json "$state_dir/comparison.json" \
        --output-md "$state_dir/comparison.md" \
        >> "$state_dir/controller.log" 2>&1 \
        || fail "Could not summarize causal-strength results"
      touch "$state_dir/complete"
      log "Causal-strength branch comparison complete"
      exit 0
    fi
    break
  done
  printf '%s\n' "$status" > "$state_dir/${active_experiment}.terminal_status.txt"
  # Preserve successful siblings before retrying only terminal non-pass jobs.
  # AMLT appends the experiment name under the output root, so recovery-wave
  # results and earlier successes remain distinct and are summarized together.
  download_passed "$active_experiment" \
    || fail "Could not download passed causal-strength results from $active_experiment"
  amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
  amlt logs view -n 1200 "$active_experiment" > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]), sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "Terminal branch state has no recoverable failed jobs"
  (( recovery_wave < 3 )) || fail "Causal-strength branches exhausted three recovery waves"
  wait_quota "$((4 * ${#failed_jobs[@]}))"
  next_wave=$((recovery_wave + 1))
  active_experiment="recurrent-m4-causal-strength-recovery${next_wave}-seed42-20260731"
  selectors=()
  for job in "${failed_jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$state_dir/recovery${next_wave}.dump.yaml" \
    || { flock -u 8; fail "Recovery AMLT dump failed"; }
  python scripts/validate_m4_causal_strength_amlt_config.py \
    --config "$state_dir/recovery${next_wave}.dump.yaml" \
    --jobs "${failed_jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "Recovery preflight failed"; }
  amlt run "$config" "${selectors[@]}" "$active_experiment" -t Feeds -y \
    --description "Resume failed M4 causal-strength branches from atomic remote outputs; recovery=$next_wave" \
    > "$state_dir/recovery${next_wave}.submission.txt" 2>&1 \
    || { flock -u 8; fail "Recovery submission failed"; }
  flock -u 8
  expected_jobs="${#failed_jobs[@]}"
done
