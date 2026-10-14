#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
train_config="amlt/feeds_recurrent_student_m4_full_20260730.yaml"
eval_config="amlt/feeds_recurrent_student_m4_eval_20260730.yaml"
m3_summary="$project_dir/results/recurrent_m3_monitor_20260730/summary.json"
m3_attention="$project_dir/results/recurrent_m3_monitor_20260730/needs_attention"
m3_scientific_failed="$project_dir/results/recurrent_m3_monitor_20260730/scientific_gate_failed"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730"
causal_rescue_marker="$state_dir/causal_rescue_in_progress"
expected_tenant="72f988bf-86f1-41af-91ab-2d7cd011db47"
expected_subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
expected_workspace_name="CS-NewsAndFeeds-Singularity"
expected_workspace_resource_group="rg-cs-newsandfeeds-singularity"
mkdir -p "$state_dir"
rm -f "$state_dir/needs_attention"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
cd "$project_dir"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"
}

fail() {
  log "ERROR: $*"
  touch "$state_dir/needs_attention"
  exit 1
}

deadline=$(( $(date +%s) + 1209600 ))
while [[ ! -f "$m3_summary" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  [[ ! -f "$m3_attention" ]] || fail "M3 execution needs attention"
  [[ ! -f "$m3_scientific_failed" ]] || fail "M3 scientific gate failed; rescue is required before M4"
  (( $(date +%s) < deadline )) || fail "M3 wait exceeded 14 days"
  sleep 120
done

readarray -t selection < <(python - "$m3_summary" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
strict = value.get("scientific_gate_pass") is True
rescue = value.get("downstream_rescue_authorized") is True
assert strict or rescue, value
arm = value.get("selected_arm")
assert arm in {"generation_only", "generation_kd"}, value
projection = value.get("selected_projection_mode")
assert projection in {"identity", "trainable"}, value
print(arm)
print(projection)
print(value.get("selection_mode"))
PY
) || fail "M3 selection is invalid"
arm="${selection[0]}"
projection="${selection[1]}"
selection_mode="${selection[2]}"
printf '%s\n' "$arm" > "$state_dir/selected_arm.txt"
printf '%s\n' "$projection" > "$state_dir/selected_projection.txt"
printf '%s\n' "$selection_mode" > "$state_dir/selection_mode.txt"
log "selected arm=$arm projection=$projection mode=$selection_mode"

# A frozen checkpoint curve can disqualify the original objective while the
# causal-rescue controller is already running a replacement curriculum.  In
# that case this legacy controller must not keep submitting recovery waves for
# a scientifically superseded run.  The rescue controller owns publication of
# pilot_complete after its full training and formal evaluation gates pass.
if [[ -f "$causal_rescue_marker" ]]; then
  log "causal-rescue controller has superseded the original M4 objective; standing down"
  touch "$state_dir/original_pilot_superseded_by_causal_rescue"
  exit 0
fi

account=""
while (( $(date +%s) < deadline )); do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  if ! az account set --subscription "$expected_subscription" >/dev/null 2>&1; then
    sleep 60
    continue
  fi
  account="$(az account show --query '{tenant:tenantId,subscription:id,state:state}' -o json 2>/dev/null || true)"
  [[ -n "$account" ]] || { sleep 60; continue; }
  python - "$account" "$expected_tenant" "$expected_subscription" <<'PY' || fail "Azure account drift"
import json, sys
value = json.loads(sys.argv[1])
assert value["tenant"] == sys.argv[2], value
assert value["subscription"] == sys.argv[3], value
assert str(value["state"]).lower() == "enabled", value
PY
  break
done
[[ -n "$account" ]] || fail "Authentication did not recover"

project_output="$(amlt project 2>&1)" || fail "$project_output"
workspace_output="$(amlt workspace list --no-update 2>&1)" || fail "$workspace_output"
grep -Fq 'project v-yuejiali' <<< "$project_output" || fail "AMLT project drift"
python scripts/feeds_workspace_utils.py \
  --name "$expected_workspace_name" \
  --resource-group "$expected_workspace_resource_group" \
  <<< "$workspace_output" || fail "AMLT workspace drift"

wait_quota() {
  local required="$1"
  local label="$2"
  while true; do
    local quota_output
    quota_output="$(amlt --json-tables ti sing -t Feeds 2>&1)" || {
      log "quota query failed for $label; retrying"
      sleep 120
      continue
    }
    if python - "$quota_output" "$state_dir/quota_${label}.json" "$required" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
text = sys.argv[1]
row = select_a100_80gb(parse_json_table(text))
quotas = row["quotas_(used/limit)"]
standard = quotas["standard"]
user = quotas["user_max"]
required = int(sys.argv[3])
available = int(standard["limit"]) - int(standard["used"])
user_available = int(user["limit"]) - int(user.get("used") or 0)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump({
        "selected_row": row,
        "required_units": required,
        "standard_available": available,
        "user_available": user_available,
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
assert available >= required and user_available >= required, {
    "required": required, "available": available, "user_available": user_available
}
PY
    then
      log "quota ready for $label ($required x 80G1-A100)"
      return 0
    fi
    log "quota insufficient for $label; retrying"
    sleep 120
  done
}

experiment_class() {
  local experiment="$1"
  local output rc
  output="$(amlt status --no-update-db "$experiment" 2>&1)"
  rc=$?
  python - "$rc" "$output" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
}

submit_selected() {
  local config="$1"
  local job="$2"
  local experiment="$3"
  local description="$4"
  local validator="$5"
  local receipt="$state_dir/${experiment}.receipt.txt"
  [[ ! -f "$receipt" ]] || return 0
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  local dump="$state_dir/${experiment}.dump.yaml"
  amlt run "$config" ":$job" --dump -t Feeds > "$dump" 2>&1 || {
    flock -u 8
    fail "AMLT dump failed for $experiment"
  }
  if [[ "$validator" == m4 ]]; then
    python scripts/validate_m4_amlt_config.py --config "$dump" --jobs "$job" \
      >> "$state_dir/controller.log" 2>&1 || {
      flock -u 8
      fail "M4 config validation failed for $job"
    }
  else
    python scripts/validate_m4_eval_amlt_config.py --config "$dump" \
      >> "$state_dir/controller.log" 2>&1 || {
      flock -u 8
      fail "M4 eval config validation failed"
    }
  fi
  local class
  class="$(experiment_class "$experiment")" || {
    flock -u 8
    fail "Could not classify experiment $experiment"
  }
  if [[ "$class" == missing ]]; then
    local output rc
    output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y --description "$description" 2>&1)"
    rc=$?
    printf '[%s] submit %s rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$rc" "$output" \
      >> "$state_dir/controller.log"
    (( rc == 0 )) || {
      flock -u 8
      fail "Submission failed for $experiment"
    }
    printf 'experiment=%s\njob=%s\nstatus=submitted\n%s\n' "$experiment" "$job" "$output" > "$receipt"
  elif [[ "$class" == exists ]]; then
    printf 'experiment=%s\njob=%s\nstatus=preexisting_not_resubmitted\n' "$experiment" "$job" > "$receipt"
  else
    flock -u 8
    fail "Ambiguous preexisting state for $experiment"
  fi
  flock -u 8
}

submit_eval() {
  local experiment="$1"
  local receipt="$state_dir/${experiment}.receipt.txt"
  [[ ! -f "$receipt" ]] || return 0
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  local dump="$state_dir/${experiment}.dump.yaml"
  amlt run "$eval_config" --dump -t Feeds > "$dump" 2>&1 || {
    flock -u 8
    fail "AMLT dump failed for $experiment"
  }
  python scripts/validate_m4_eval_amlt_config.py --config "$dump" \
    >> "$state_dir/controller.log" 2>&1 || {
    flock -u 8
    fail "M4 eval config validation failed"
  }
  local class
  class="$(experiment_class "$experiment")" || {
    flock -u 8
    fail "Could not classify experiment $experiment"
  }
  if [[ "$class" == missing ]]; then
    local output rc
    output="$(amlt run "$eval_config" "$experiment" -t Feeds -y \
      --description 'Formal T1 T2 T3 T5 T6 and frozen recurrent C4 evaluation of selected M4 checkpoint' 2>&1)"
    rc=$?
    printf '[%s] submit %s rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$rc" "$output" \
      >> "$state_dir/controller.log"
    (( rc == 0 )) || {
      flock -u 8
      fail "Submission failed for $experiment"
    }
    printf 'experiment=%s\nstatus=submitted\n%s\n' "$experiment" "$output" > "$receipt"
  elif [[ "$class" == exists ]]; then
    printf 'experiment=%s\nstatus=preexisting_not_resubmitted\n' "$experiment" > "$receipt"
  else
    flock -u 8
    fail "Ambiguous preexisting state for $experiment"
  fi
  flock -u 8
}

wait_pass() {
  local experiment="$1"
  local expected_jobs="$2"
  local stage="$3"
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
    local status
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
      >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
      sleep 120
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$expected_jobs" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[2], int(sys.argv[1])) else 1)
PY
    then
      log "$stage passed"
      printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
      return 0
    fi
    printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
    amlt logs view -n 800 "$experiment" > "$state_dir/${experiment}.failure.log" 2>&1 || true
    fail "$stage did not pass"
  done
}

wait_pass_recoverable() {
  local experiment="$1"
  local expected_jobs="$2"
  local stage="$3"
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
    local status
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
      >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
      sleep 120
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$expected_jobs" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[2], int(sys.argv[1])) else 1)
PY
    then
      log "$stage passed"
      printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
      return 0
    fi
    printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
    amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
    amlt logs view -n 1200 "$experiment" > "$state_dir/${experiment}.failure.log" 2>&1 || true
    log "$stage attempt failed: $experiment"
    return 1
  done
}

cache_experiment="recurrent-m4-cache-seed42-fixedu4-v2-20260730"
diagnostic_job="m4-diagnostic-${arm//_/-}-${projection}-seed42"
diagnostic_experiment="recurrent-m4-diagnostic-${arm//_/-}-${projection}-seed42-fixedu4-v2-20260730"
full_job="m4-full-${arm//_/-}-${projection}-seed42"
full_experiment="${M4_FULL_EXPERIMENT:-recurrent-m4-full-${arm//_/-}-${projection}-seed42-fixedu4-dist4-v7-maskce-20260731}"
eval_experiment="recurrent-m4-formal-eval-${arm//_/-}-${projection}-seed42-fixedu4-v2-20260730"

if [[ "$arm" == generation_kd ]]; then
  wait_quota 1 m4_cache
  submit_selected "$train_config" m4-full-teacher-cache-seed42 "$cache_experiment" \
    'M4 full-data teacher cache for selected behavioral-KD arm' m4
  wait_pass "$cache_experiment" 1 "M4 teacher cache"
fi

wait_quota 1 m4_diagnostic
submit_selected "$train_config" "$diagnostic_job" "$diagnostic_experiment" \
  "M4 selected diagnostic arm=$arm projection=$projection" m4
wait_pass "$diagnostic_experiment" 1 "M4 diagnostic"
diagnostic_download="$state_dir/diagnostic_download"
mkdir -p "$diagnostic_download"
amlt results download "$diagnostic_experiment" --status pass \
  --include 'completion.json' \
  --include 'gradient_audit.json' \
  --include 'validation_metrics.json' \
  --include 'causal_validation_metrics.json' \
  --include 'run_config.json' \
  --output "$diagnostic_download" >> "$state_dir/controller.log" 2>&1 \
  || fail "Could not download M4 diagnostic"
mapfile -t diagnostic_completions < <(find "$diagnostic_download" -type f -path '*/metrics/completion.json')
[[ "${#diagnostic_completions[@]}" -eq 1 ]] || fail "Expected one diagnostic completion file"
diagnostic_root="$(dirname "${diagnostic_completions[0]}")"
python scripts/validate_m4_diagnostic_result.py \
  --root "$diagnostic_root" \
  --projection "$projection" \
  --arm "$arm" \
  --output "$state_dir/diagnostic_gate.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "M4 diagnostic scientific/engineering gate failed"

full_base_experiment="$full_experiment"
full_passed=false
for recovery_wave in 0 1 2 3; do
  if [[ -f "$causal_rescue_marker" ]]; then
    log "causal-rescue controller has superseded the original M4 objective; standing down"
    touch "$state_dir/original_pilot_superseded_by_causal_rescue"
    exit 0
  fi
  if (( recovery_wave > 0 )); then
    full_experiment="${full_base_experiment}-recovery${recovery_wave}"
  fi
  wait_quota 4 "m4_full_recovery${recovery_wave}"
  submit_selected "$train_config" "$full_job" "$full_experiment" \
    "M4 selected one-epoch pilot arm=$arm projection=$projection with causal grounding; recovery_wave=$recovery_wave resumes latest atomic checkpoint" m4
  if wait_pass_recoverable "$full_experiment" 1 "M4 one-epoch pilot"; then
    full_passed=true
    break
  fi
done
[[ "$full_passed" == true ]] || fail "M4 one-epoch pilot exhausted three recovery waves"
full_download="$state_dir/full_download"
mkdir -p "$full_download"
amlt results download "$full_experiment" --status pass \
  --include 'completion.json' \
  --include 'gradient_audit.json' \
  --include 'validation_metrics.json' \
  --include 'causal_validation_metrics.json' \
  --include 'training_metrics.json' \
  --include 'run_config.json' \
  --output "$full_download" >> "$state_dir/controller.log" 2>&1 \
  || fail "Could not download M4 full metrics"

eval_base_experiment="$eval_experiment"
eval_passed=false
for recovery_wave in 0 1 2 3; do
  if (( recovery_wave > 0 )); then
    eval_experiment="${eval_base_experiment}-recovery${recovery_wave}"
  fi
  wait_quota 5 "m4_eval_recovery${recovery_wave}"
  submit_eval "$eval_experiment"
  if wait_pass_recoverable "$eval_experiment" 5 "M4 formal evaluation"; then
    eval_passed=true
    break
  fi
done
[[ "$eval_passed" == true ]] || fail "M4 formal evaluation exhausted three recovery waves"
eval_download="$state_dir/eval_download"
mkdir -p "$eval_download"
amlt results download "$eval_experiment" --status pass \
  --include 'metrics.json' \
  --include 'recurrent_causal_gate.json' \
  --include 'predictions.jsonl' \
  --include 'predictions_redacted.jsonl' \
  --include 'manifest_audit.json' \
  --include 'selection.txt' \
  --output "$eval_download" >> "$state_dir/controller.log" 2>&1 \
  || fail "Could not download M4 evaluation"

python scripts/validate_m4_pilot_result.py \
  --root "$eval_download" \
  --output "$state_dir/pilot_scientific_gate.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "M4 pilot task/causal scientific gate failed"

touch "$state_dir/pilot_complete"
log "M4 pilot and formal evaluation passed; final full-corpus fit is authorized"
