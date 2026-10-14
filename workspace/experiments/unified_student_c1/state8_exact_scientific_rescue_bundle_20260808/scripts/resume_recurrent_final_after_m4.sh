#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
fit_config="amlt/feeds_recurrent_student_final_fit_20260730.yaml"
eval_config="amlt/feeds_recurrent_student_final_eval_20260730.yaml"
m3_summary="$project_dir/results/recurrent_m3_monitor_20260730/summary.json"
m3_attention="$project_dir/results/recurrent_m3_monitor_20260730/needs_attention"
m3_scientific_failed="$project_dir/results/recurrent_m3_monitor_20260730/scientific_gate_failed"
m4_dir="$project_dir/results/recurrent_m4_pipeline_20260730"
causal_rescue_marker="$m4_dir/causal_rescue_in_progress"
causal_rescue_attention="$m4_dir/causal_rescue/needs_attention"
state_dir="$project_dir/results/recurrent_final_pipeline_20260730"
cache_v2_root="/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way"
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

heartbeat() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
}

deadline=$(( $(date +%s) + 2592000 ))

wait_for_selection() {
  while [[ ! -f "$m3_summary" ]]; do
    heartbeat
    [[ ! -f "$m3_attention" ]] || fail "M3 execution needs attention"
    [[ ! -f "$m3_scientific_failed" ]] || fail "M3 scientific gate failed; rescue is required before final fit"
    (( $(date +%s) < deadline )) || fail "M3 selection wait exceeded 30 days"
    sleep 120
  done
  readarray -t selection < <(python - "$m3_summary" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
strict = value.get("scientific_gate_pass") is True
rescue = value.get("downstream_rescue_authorized") is True
assert strict or rescue, value
assert value.get("selected_arm") in {"generation_only", "generation_kd"}, value
assert value.get("selected_projection_mode") in {"identity", "trainable"}, value
print(value["selected_arm"])
print(value["selected_projection_mode"])
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
}

wait_for_auth() {
  account=""
  while (( $(date +%s) < deadline )); do
    heartbeat
    if ! az account set --subscription "$expected_subscription" >/dev/null 2>&1; then
      sleep 60
      continue
    fi
    account="$(az account show --query '{tenant:tenantId,subscription:id,state:state}' -o json 2>/dev/null || true)"
    [[ -n "$account" ]] || { sleep 60; continue; }
    if python - "$account" "$expected_tenant" "$expected_subscription" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
assert value["tenant"] == sys.argv[2], value
assert value["subscription"] == sys.argv[3], value
assert str(value["state"]).lower() == "enabled", value
PY
    then
      break
    fi
    fail "Authenticated Azure account drifted from fixed Feeds subscription"
  done
  [[ -n "$account" ]] || fail "Azure authentication did not recover"
  project_output="$(amlt project 2>&1)" || fail "$project_output"
  workspace_output="$(amlt workspace list --no-update 2>&1)" || fail "$workspace_output"
  grep -Fq 'project v-yuejiali' <<< "$project_output" || fail "AMLT project drift"
  python scripts/feeds_workspace_utils.py \
    --name "$expected_workspace_name" \
    --resource-group "$expected_workspace_resource_group" \
    <<< "$workspace_output" || fail "AMLT workspace drift"
  log "fixed Feeds account/project/workspace verified"
}

wait_quota() {
  local required="$1"
  local label="$2"
  while true; do
    heartbeat
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
required = int(sys.argv[3])
available = int(quotas["standard"]["limit"]) - int(quotas["standard"]["used"])
user_available = int(quotas["user_max"]["limit"]) - int(quotas["user_max"].get("used") or 0)
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
      log "quota ready for $label ($required A100-80GB units)"
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

submit_selected_fit() {
  local experiment="$1"
  shift
  local jobs=("$@")
  local receipt="$state_dir/${experiment}.receipt.txt"
  [[ ! -f "$receipt" ]] || return 0
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  local dump="$state_dir/${experiment}.dump.yaml"
  local selectors=()
  local job
  for job in "${jobs[@]}"; do selectors+=(":$job"); done
  amlt run "$fit_config" "${selectors[@]}" --dump -t Feeds > "$dump" 2>&1 || {
    flock -u 8
    fail "AMLT dump failed for $experiment"
  }
  python scripts/validate_final_fit_amlt_config.py --config "$dump" --jobs "${jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 || {
    flock -u 8
    fail "Final-fit config validation failed for $experiment"
  }
  local class
  class="$(experiment_class "$experiment")" || {
    flock -u 8
    fail "Could not classify experiment $experiment"
  }
  if [[ "$class" == missing ]]; then
    local output rc
    output="$(amlt run "$fit_config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Strict recurrent unified final-fit stage: ${jobs[*]}" 2>&1)"
    rc=$?
    printf '[%s] submit %s rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$rc" "$output" \
      >> "$state_dir/controller.log"
    (( rc == 0 )) || {
      flock -u 8
      fail "Submission failed for $experiment"
    }
    printf 'experiment=%s\njobs=%s\nstatus=submitted\n%s\n' "$experiment" "${jobs[*]}" "$output" > "$receipt"
  elif [[ "$class" == exists ]]; then
    printf 'experiment=%s\njobs=%s\nstatus=preexisting_not_resubmitted\n' "$experiment" "${jobs[*]}" > "$receipt"
  else
    flock -u 8
    fail "Ambiguous pre-existing state for $experiment"
  fi
  flock -u 8
}

submit_final_eval() {
  local experiment="$1"
  local receipt="$state_dir/${experiment}.receipt.txt"
  [[ ! -f "$receipt" ]] || return 0
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  local dump="$state_dir/${experiment}.dump.yaml"
  amlt run "$eval_config" --dump -t Feeds > "$dump" 2>&1 || {
    flock -u 8
    fail "Final-eval AMLT dump failed"
  }
  python scripts/validate_final_eval_amlt_config.py --config "$dump" \
    >> "$state_dir/controller.log" 2>&1 || {
    flock -u 8
    fail "Final-eval config validation failed"
  }
  local class
  class="$(experiment_class "$experiment")" || {
    flock -u 8
    fail "Could not classify experiment $experiment"
  }
  if [[ "$class" == missing ]]; then
    local output rc
    output="$(amlt run "$eval_config" "$experiment" -t Feeds -y \
      --description 'Final strict recurrent T1 T2 T3 T5 T6 and frozen 150-pair C4 evaluation' 2>&1)"
    rc=$?
    printf '[%s] submit %s rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$rc" "$output" \
      >> "$state_dir/controller.log"
    (( rc == 0 )) || {
      flock -u 8
      fail "Final evaluation submission failed"
    }
    printf 'experiment=%s\nstatus=submitted\n%s\n' "$experiment" "$output" > "$receipt"
  elif [[ "$class" == exists ]]; then
    printf 'experiment=%s\nstatus=preexisting_not_resubmitted\n' "$experiment" > "$receipt"
  else
    flock -u 8
    fail "Ambiguous pre-existing final evaluation"
  fi
  flock -u 8
}

wait_pass() {
  local experiment="$1"
  local expected_jobs="$2"
  local stage="$3"
  while true; do
    heartbeat
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
      printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
      log "$stage passed"
      return 0
    fi
    printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
    amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
    amlt logs view -n 1200 "$experiment" > "$state_dir/${experiment}.failure.log" 2>&1 || true
    fail "$stage did not pass"
  done
}

wait_pass_recoverable() {
  local experiment="$1"
  local expected_jobs="$2"
  local stage="$3"
  while true; do
    heartbeat
    local status
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
      >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
      sleep 120
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$expected_jobs" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[2], int(sys.argv[1])) else 1)
PY
    then
      printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
      log "$stage passed"
      return 0
    fi
    printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
    amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
    amlt logs view -n 1200 "$experiment" > "$state_dir/${experiment}.failure.log" 2>&1 || true
    log "$stage attempt failed: $experiment"
    return 1
  done
}

wait_for_pilot_gate() {
  # ``pilot_complete`` can be published by the original M4 while a separately
  # authorized causal-rescue controller is still selecting or validating a
  # replacement.  Do not let final fit consume a receipt that may be promoted
  # again underneath it; the rescue owner clears its marker atomically only
  # after it either stands down or finishes receipt promotion.
  while [[ ! -f "$m4_dir/pilot_complete" || -f "$causal_rescue_marker" ]]; do
    heartbeat
    if [[ -f "$m4_dir/pilot_complete" && -f "$causal_rescue_marker" ]]; then
      [[ ! -f "$causal_rescue_attention" ]] \
        || fail "causal-rescue needs attention after pilot publication"
      log "pilot is complete; waiting for causal-rescue receipt ownership to settle"
      sleep 120
      continue
    fi
    if [[ -f "$m4_dir/needs_attention" ]]; then
      if [[ -f "$causal_rescue_marker" && ! -f "$causal_rescue_attention" ]]; then
        log "original M4 needs attention; causal-rescue controller is active"
      else
        fail "M4 pilot pipeline needs attention"
      fi
    fi
    (( $(date +%s) < deadline )) || fail "M4 pilot wait exceeded 30 days"
    sleep 120
  done
  python - "$m4_dir/pilot_scientific_gate.json" <<'PY' || fail "M4 pilot scientific receipt is invalid"
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value.get("pass") is True, value
assert value.get("stage") == "M4_strict_recurrent_pilot", value
PY
  log "M4 task/causal scientific gate verified"
}

select_final_fit_topology() {
  local diagnostic="recurrent-multinode-dist-diag-16rank-seed42-20260731"
  local download="$state_dir/multinode_dist_diag_download"
  while true; do
    heartbeat
    local status
    status="$(amlt status "$diagnostic" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$diagnostic" "$status" \
      >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
      sleep 120
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    then
      rm -rf "$download"
      mkdir -p "$download"
      if amlt results download "$diagnostic" --status pass --include 'rank-*.json' \
          --output "$download" >> "$state_dir/controller.log" 2>&1 \
        && python scripts/validate_multinode_dist_diag.py --root "$download" \
          --output "$state_dir/multinode_dist_diag_gate.json" \
          >> "$state_dir/controller.log" 2>&1; then
        printf '%s\n' 16 > "$state_dir/selected_finalfit_world_size.txt"
        printf '%s\n' 1 > "$state_dir/selected_finalfit_gradient_accumulation.txt"
        log "verified two-node 16-rank topology selected; exact global batch remains 16"
        return 0
      fi
      log "16-rank diagnostic receipts failed validation; falling back to audited 8-rank topology"
    else
      amlt show "$diagnostic" > "$state_dir/multinode_dist_diag.show.txt" 2>&1 || true
      amlt logs view -n 600 "$diagnostic" > "$state_dir/multinode_dist_diag.failure.log" 2>&1 || true
      log "16-rank diagnostic did not pass; falling back to audited 8-rank topology"
    fi
    printf '%s\n' 8 > "$state_dir/selected_finalfit_world_size.txt"
    printf '%s\n' 2 > "$state_dir/selected_finalfit_gradient_accumulation.txt"
    return 0
  done
}

wait_for_selection
wait_for_auth

cache_experiment="recurrent-finalfit-cache-${arm//_/-}-${projection}-seed42-v1-20260730"
if [[ "$arm" == generation_kd ]]; then
  cache_jobs=(
    finalfit-cache-rank0-seed42
    finalfit-cache-rank1-seed42
    finalfit-cache-rank2-seed42
    finalfit-cache-rank3-seed42
  )
  if [[ -f "$state_dir/cache_v2_16way/complete" ]]; then
    log "16-way exact teacher cache is already complete; legacy 4-way cache skipped"
  elif [[ -f "$state_dir/cache_v2_16way/controller.active.pid" ]] \
    && kill -0 "$(cat "$state_dir/cache_v2_16way/controller.active.pid")" 2>/dev/null \
    && [[ ! -f "$state_dir/cache_v2_16way/needs_attention" ]]; then
    log "16-way exact teacher cache controller is healthy; legacy 4-way cache skipped"
  else
    log "16-way cache controller is unavailable; legacy 4-way fallback will be prepared"
  fi
fi

wait_for_pilot_gate

if [[ "$arm" == generation_kd ]]; then
  # The exact 16-way cache uses the same batch-1/eager numeric path but is
  # independently sharded for throughput. Its monitor publishes a verified
  # merged receipt atomically. Fall back to the original four-way cache only
  # when the v2 controller explicitly reports terminal failure.
  cache_v2_failed=false
  while [[ ! -f "$state_dir/cache_v2_16way/complete" ]]; do
    heartbeat
    if [[ -f "$state_dir/cache_v2_16way/needs_attention" ]]; then
      cache_v2_failed=true
      log "16-way cache exhausted targeted recovery; activating 4-way fallback"
      break
    fi
    (( $(date +%s) < deadline )) || fail "Final-fit 16-way cache wait exceeded 30 days"
    sleep 120
  done
  if [[ "$cache_v2_failed" == false ]]; then
    log "verified 16-way exact teacher cache selected"
  else
    legacy_cache_passed=false
    for recovery_wave in 1 2 3; do
      legacy_cache_experiment="${cache_experiment}-recovery${recovery_wave}-20260731"
      wait_quota 4 "legacy_cache_recovery${recovery_wave}"
      submit_selected_fit "$legacy_cache_experiment" "${cache_jobs[@]}"
      if wait_pass_recoverable "$legacy_cache_experiment" 4 "Legacy 4-way final-fit cache"; then
        legacy_cache_passed=true
        break
      fi
    done
    [[ "$legacy_cache_passed" == true ]] \
      || fail "16-way and legacy 4-way teacher caches exhausted recovery waves"
    log "verified legacy 4-way exact teacher cache fallback"
  fi
fi

select_final_fit_topology
fit_world_size="$(cat "$state_dir/selected_finalfit_world_size.txt")"
fit_job="finalfit-${arm//_/-}-${projection}-seed42"
fit_experiment="recurrent-finalfit-${arm//_/-}-${projection}-seed42-full259845-v1-20260730"
if [[ "$fit_world_size" == 16 ]]; then
  fit_job="${fit_job}-16rank"
  fit_experiment="recurrent-finalfit-${arm//_/-}-${projection}-seed42-full259845-v2-16rank-20260731"
elif [[ "$fit_world_size" != 8 ]]; then
  fail "Unknown selected final-fit world size: $fit_world_size"
fi
fit_base_experiment="$fit_experiment"
fit_passed=false
for recovery_wave in 0 1 2 3; do
  if (( recovery_wave > 0 )); then
    fit_experiment="${fit_base_experiment}-recovery${recovery_wave}"
  fi
  wait_quota "$fit_world_size" "finalfit_train_recovery${recovery_wave}"
  submit_selected_fit "$fit_experiment" "$fit_job"
  if wait_pass_recoverable "$fit_experiment" 1 "259845-row final fit"; then
    fit_passed=true
    break
  fi
done
[[ "$fit_passed" == true ]] || fail "259845-row final fit exhausted three recovery waves"

fit_download="$state_dir/final_fit_download"
mkdir -p "$fit_download"
amlt results download "$fit_experiment" --status pass \
  --include 'adapter_model.safetensors' \
  --include 'adapter_config.json' \
  --include 'recurrent_state.pt' \
  --include 'completion.json' \
  --include 'validation_metrics.json' \
  --include 'causal_validation_metrics.json' \
  --include 'training_metrics.json' \
  --include 'run_config.json' \
  --include 'gradient_audit.json' \
  --include 'data_audit.json' \
  --include 'final_fit_manifest.json' \
  --include 'pilot_receipt.json' \
  --output "$fit_download" >> "$state_dir/controller.log" 2>&1 \
  || fail "Could not download final-fit artifact"
python scripts/validate_final_fit_result.py \
  --root "$fit_download" \
  --arm "$arm" \
  --projection "$projection" \
  --output "$state_dir/final_fit_gate.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "Final-fit artifact/exposure gate failed"
touch "$state_dir/final_fit_complete"

eval_experiment="recurrent-final-eval-${arm//_/-}-${projection}-seed42-full259845-v1-20260730"
eval_base_experiment="$eval_experiment"
eval_passed=false
for recovery_wave in 0 1 2 3; do
  if (( recovery_wave > 0 )); then
    eval_experiment="${eval_base_experiment}-recovery${recovery_wave}"
  fi
  wait_quota 5 "final_eval_recovery${recovery_wave}"
  submit_final_eval "$eval_experiment"
  if wait_pass_recoverable "$eval_experiment" 5 "Final formal/C4 evaluation"; then
    eval_passed=true
    break
  fi
done
[[ "$eval_passed" == true ]] || fail "Final formal/C4 evaluation exhausted three recovery waves"

eval_download="$state_dir/final_eval_download"
mkdir -p "$eval_download"
amlt results download "$eval_experiment" --status pass \
  --include 'metrics.json' \
  --include 'recurrent_causal_gate.json' \
  --include 'predictions.jsonl' \
  --include 'predictions_redacted.jsonl' \
  --include 'manifest_audit.json' \
  --include 'selection.txt' \
  --output "$eval_download" >> "$state_dir/controller.log" 2>&1 \
  || fail "Could not download final formal/C4 evaluation"
python scripts/validate_m4_pilot_result.py \
  --root "$eval_download" \
  --output "$state_dir/final_scientific_gate_without_t3_semantic.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "Final task/C4 scientific gate failed"

touch "$state_dir/final_eval_complete"
touch "$state_dir/t3_semantic_judge_pending"
log "Final fit and strict formal/C4 evaluation complete; entering T3 semantic judge"
bash scripts/resume_recurrent_final_t3_judge.sh \
  || fail "Final T3 semantic judge controller failed"
test -f "$state_dir/overall_complete" \
  || fail "Final aggregate completion marker is missing after T3 judge"
log "Complete unified recurrent model pipeline passed all aggregate gates"
