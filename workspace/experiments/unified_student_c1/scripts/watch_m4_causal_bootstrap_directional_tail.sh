#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_directional_tail"
config="$project_dir/amlt/feeds_m4_causal_bootstrap_directional_tail_20260731.yaml"
base_experiment="recurrent-m4-causal-bootstrap-directional-tail-seed42-20260731"
expected_tenant="72f988bf-86f1-41af-91ab-2d7cd011db47"
expected_subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
jobs=(
  m4-causal-bootstrap-balanced-sum-w1-round0to500-seed42
  m4-causal-bootstrap-balanced-sum-w4-round0to500-seed42
)
expected_branches=(bootstrap_balanced_sum_w1 bootstrap_balanced_sum_w4)
mkdir -p "$state_dir/download"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "$state_dir/controller.log"
}
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

account_matches() {
  python -c '
import json, sys
x = json.loads(sys.argv[1])
assert x["tenant"] == sys.argv[2] and x["subscription"] == sys.argv[3], x
assert str(x["state"]).lower() == "enabled" and x["type"] == "user", x
' "$1" "$expected_tenant" "$expected_subscription" >/dev/null 2>&1
}

wait_auth() {
  local account project_output
  while true; do
    heartbeat
    account="$(az account show --query '{tenant:tenantId,subscription:id,state:state,type:user.type}' -o json 2>/dev/null || true)"
    if ! account_matches "$account"; then
      az account set --subscription "$expected_subscription" >/dev/null 2>&1 \
        || { sleep 60; continue; }
      account="$(az account show --query '{tenant:tenantId,subscription:id,state:state,type:user.type}' -o json 2>/dev/null || true)"
    fi
    account_matches "$account" || { sleep 60; continue; }
    project_output="$(amlt project 2>&1)" || { sleep 60; continue; }
    grep -Eq '(^|[[:space:]])(project|PROJECT_NAME)[[:space:]]+v-yuejiali([[:space:]]|$)' \
      <<< "$project_output" || { sleep 60; continue; }
    return 0
  done
}

wait_quota() {
  local required="$1"
  while true; do
    heartbeat
    local quota
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" "$state_dir/quota_before_recovery.json" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1]))
q = row['quotas_(used/limit)']; required = int(sys.argv[2])
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

download_passed() {
  local experiment="$1" status
  status="$(amlt status "$experiment" 2>&1)" || return 1
  grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status" || return 0
  amlt results download "$experiment" --status pass \
    --include 'branch_training_summary.json' \
    --include 'validation_metrics.json' \
    --include 'causal_validation_metrics.json' \
    --include 'training_metrics.json' \
    --include 'run_config.json' \
    --include 'recurrent_causal_gate.json' \
    --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1
}

submit_if_missing() {
  local experiment="$1"; shift
  local selected=("$@") selectors=() item status rc class output
  for item in "${selected[@]}"; do selectors+=(":$item"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.dump.yaml" 2>&1 \
    || { flock -u 8; exec 8>&-; fail "directional-tail AMLT dump failed"; }
  python scripts/validate_m4_causal_bootstrap_rehearsal_amlt_config.py \
    --config "$state_dir/${experiment}.dump.yaml" --jobs "${selected[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; exec 8>&-; fail "directional-tail preflight failed"; }
  status="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || { flock -u 8; exec 8>&-; fail "could not classify directional-tail experiment"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Recovery for donor-bootstrap directional-tail causal curriculum; jobs=${selected[*]}" 2>&1)"
    rc=$?
    printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; exec 8>&-; fail "directional-tail submission failed"; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; exec 8>&-; fail "ambiguous directional-tail experiment state"
  fi
  flock -u 8
  exec 8>&-
}

python scripts/validate_m4_causal_bootstrap_rehearsal_amlt_config.py \
  --config "$config" --jobs "${jobs[@]}" >> "$state_dir/controller.log" 2>&1 \
  || fail "directional-tail config validation failed"

active_experiment="$base_experiment"
active_jobs=("${jobs[@]}")
wait_auth
for recovery_wave in 0 1 2 3; do
  submit_if_missing "$active_experiment" "${active_jobs[@]}"
  while true; do
    heartbeat
    status="$(amlt status "$active_experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$active_experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
      wait_auth
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$status" "${#active_jobs[@]}" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
    then
      printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
      download_passed "$active_experiment" || fail "directional-tail download failed"
      python scripts/summarize_m4_causal_strength_branches.py \
        --root "$state_dir/download" \
        --expected-branches "${expected_branches[@]}" \
        --output-json "$state_dir/comparison.json" \
        --output-md "$state_dir/comparison.md" \
        >> "$state_dir/controller.log" 2>&1 \
        || fail "directional-tail comparison failed"
      touch "$state_dir/complete"
      log "directional-tail comparison complete"
      exit 0
    fi
    break
  done
  printf '%s\n' "$status" > "$state_dir/${active_experiment}.terminal_status.txt"
  download_passed "$active_experiment" || fail "could not preserve passed directional-tail sibling"
  amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
  timeout 180s amlt logs view -n 1200 "$active_experiment" \
    > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]), sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal directional-tail state has no recoverable jobs"
  (( recovery_wave < 3 )) || fail "directional-tail exhausted recovery waves"
  wait_quota "$((4 * ${#failed_jobs[@]}))"
  wait_auth
  active_experiment="recurrent-m4-causal-bootstrap-directional-tail-recovery$((recovery_wave + 1))-seed42-20260731"
  active_jobs=("${failed_jobs[@]}")
done
