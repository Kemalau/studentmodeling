#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
m4_dir="$project_dir/results/recurrent_m4_pipeline_20260730"
branch_dir="$m4_dir/causal_strength_branches"
rehearsal_dir="$m4_dir/causal_rehearsal"
rehearsal_complete="$m4_dir/causal_rehearsal/complete"
rehearsal_early_stopped="$m4_dir/causal_rehearsal/early_stopped.json"
sum_rehearsal_dir="$m4_dir/causal_rehearsal_sum"
sum_rehearsal_complete="$sum_rehearsal_dir/complete"
sum_rehearsal_early_stopped="$sum_rehearsal_dir/early_stopped.json"
bootstrap_rehearsal_dir="$m4_dir/causal_bootstrap_rehearsal"
bootstrap_rehearsal_complete="$bootstrap_rehearsal_dir/complete"
bootstrap_rehearsal_early_stopped="$bootstrap_rehearsal_dir/early_stopped.json"
bootstrap_directional_dir="$m4_dir/causal_bootstrap_directional_tail"
bootstrap_directional_complete="$bootstrap_directional_dir/complete"
donor_directional_dir="$m4_dir/causal_donor_directional"
donor_directional_complete="$donor_directional_dir/complete"
cf4_dir="$m4_dir/cf4_diagnostic"
cf4_complete="$cf4_dir/complete"
cf4_attention="$cf4_dir/needs_attention"
escalation_dir="$m4_dir/causal_strength_escalation"
state_dir="$m4_dir/causal_rescue"
full_base="$project_dir/amlt/feeds_m4_causal_strength_branches_20260731.yaml"
eval_base="$project_dir/amlt/feeds_recurrent_student_m4_eval_20260730.yaml"
full_config="$project_dir/amlt/feeds_m4_causal_rescue_selected_20260731.yaml"
eval_config="$project_dir/amlt/feeds_m4_causal_rescue_eval_selected_20260731.yaml"
account="csnewsandfeeds4150361735"
container="amulet"
isolated_receipt="studentmodeling/world_model_v1/m4/causal_rescue/selected_seed42.json"
main_receipt="studentmodeling/world_model_v1/m4/selected_seed42.json"
mkdir -p "$state_dir"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
touch "$m4_dir/causal_rescue_in_progress"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "$state_dir/controller.log"
}

heartbeat() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
}

fail() {
  log "ERROR: $*"
  touch "$state_dir/needs_attention"
  exit 1
}

log "causal-rescue controller active; waiting for diagnostic families and bootstrap rehearsal"

stand_down_if_original_passed() {
  if [[ -f "$m4_dir/pilot_complete" ]]; then
    log "original M4 passed before causal-rescue full submission; standing down"
    rm -f "$m4_dir/causal_rescue_in_progress"
    touch "$state_dir/not_needed" "$state_dir/complete"
    exit 0
  fi
}

experiment_class() {
  local experiment="$1" output rc
  output="$(amlt status --no-update-db "$experiment" 2>&1)"
  rc=$?
  python - "$rc" "$output" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
}

wait_quota() {
  local required="$1" label="$2"
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" "$state_dir/quota_${label}.json" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1]))
quota = row['quotas_(used/limit)']; required = int(sys.argv[2])
available = int(quota['standard']['limit']) - int(quota['standard']['used'])
user_available = int(quota['user_max']['limit']) - int(quota['user_max'].get('used') or 0)
open(sys.argv[3], 'w', encoding='utf-8').write(json.dumps({
    'selected_row': row, 'required_units': required,
    'standard_available': available, 'user_available': user_available,
}, indent=2, sort_keys=True) + '\n')
assert available >= required and user_available >= required
PY
    then
      log "quota ready for $label ($required A100-80GB units)"
      return 0
    fi
    sleep 120
  done
}

submit_config() {
  local config="$1" experiment="$2" description="$3" validator="$4"
  shift 4
  local jobs=("$@") selectors=() job dump class output rc
  for job in "${jobs[@]}"; do selectors+=(":$job"); done
  dump="$state_dir/${experiment}.dump.yaml"
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$dump" 2>&1 \
    || { flock -u 8; fail "AMLT dump failed for $experiment"; }
  if [[ "$validator" == rescue ]]; then
    python scripts/validate_m4_causal_rescue_amlt_config.py \
      --config "$dump" --jobs "${jobs[@]}" >> "$state_dir/controller.log" 2>&1 \
      || { flock -u 8; fail "causal-rescue preflight failed for $experiment"; }
  else
    python scripts/validate_m4_eval_amlt_config.py --config "$dump" \
      >> "$state_dir/controller.log" 2>&1 \
      || { flock -u 8; fail "causal-rescue eval preflight failed"; }
  fi
  class="$(experiment_class "$experiment")" \
    || { flock -u 8; fail "could not classify $experiment"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "$description" 2>&1)"
    rc=$?
    printf '[%s] submit %s rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$rc" "$output" >> "$state_dir/controller.log"
    (( rc == 0 )) || { flock -u 8; fail "submission failed for $experiment"; }
    printf 'experiment=%s\njobs=%s\nstatus=submitted\n%s\n' \
      "$experiment" "${jobs[*]}" "$output" > "$state_dir/${experiment}.receipt.txt"
  elif [[ "$class" == exists ]]; then
    printf 'experiment=%s\njobs=%s\nstatus=preexisting_not_resubmitted\n' \
      "$experiment" "${jobs[*]}" > "$state_dir/${experiment}.receipt.txt"
  else
    flock -u 8
    fail "ambiguous pre-existing experiment $experiment"
  fi
  flock -u 8
}

wait_pass_recoverable() {
  local experiment="$1" expected="$2" stage="$3"
  while true; do
    heartbeat
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
      sleep 120
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$status" "$expected" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
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

deadline=$(( $(date +%s) + 2592000 ))
while [[ ! -f "$branch_dir/complete" \
        || ( ! -f "$rehearsal_complete" && ! -f "$rehearsal_early_stopped" ) \
        || ( ! -f "$sum_rehearsal_complete" \
             && ! -f "$sum_rehearsal_early_stopped" ) \
        || ( ! -f "$bootstrap_rehearsal_complete" \
             && ! -f "$bootstrap_rehearsal_early_stopped" \
             && ! -f "$bootstrap_rehearsal_dir/needs_attention" ) \
        || ( ! -f "$bootstrap_directional_complete" \
             && ! -f "$bootstrap_directional_dir/needs_attention" ) \
        || ( ! -f "$donor_directional_complete" \
             && ! -f "$donor_directional_dir/needs_attention" ) \
        || ( ! -f "$cf4_complete" && ! -f "$cf4_attention" ) ]]; do
  heartbeat
  [[ -f "$branch_dir/complete" || ! -f "$branch_dir/needs_attention" ]] \
    || fail "causal-strength branches need attention"
  [[ -f "$rehearsal_complete" || -f "$rehearsal_early_stopped" \
      || ! -f "$rehearsal_dir/needs_attention" ]] \
    || fail "dense causal rehearsal needs attention"
  [[ -f "$sum_rehearsal_complete" || -f "$sum_rehearsal_early_stopped" \
      || ! -f "$sum_rehearsal_dir/needs_attention" ]] \
    || fail "sequence-aligned causal rehearsal needs attention"
  # The directional-tail family is an independently submitted comparison.
  # Exhausting its infrastructure recovery must not invalidate already complete
  # candidate families; the selection block below records and excludes it.
  [[ ! -f "$m4_dir/pilot_complete" ]] || {
    log "original M4 already passed; rescue is unnecessary"
    rm -f "$m4_dir/causal_rescue_in_progress"
    touch "$state_dir/not_needed" "$state_dir/complete"
    exit 0
  }
  (( $(date +%s) < deadline )) || fail "causal diagnostic comparison wait exceeded 30 days"
  sleep 120
done

stand_down_if_original_passed

selection_json="$state_dir/candidate_selection.json"
candidate_args=(
  --candidate "sparse=$branch_dir/comparison.json"
)
if [[ -f "$sum_rehearsal_complete" && -f "$sum_rehearsal_dir/comparison.json" ]]; then
  candidate_args+=(--candidate "sum_aligned=$sum_rehearsal_dir/comparison.json")
else
  log "symmetric sequence-sum rehearsal was early-stopped; bootstrap supersedes it"
fi
if [[ -f "$rehearsal_complete" && -f "$rehearsal_dir/comparison.json" ]]; then
  candidate_args+=(--candidate "dense=$rehearsal_dir/comparison.json")
else
  log "symmetric-mean dense rehearsal was early-stopped; excluding dead objective"
fi
if [[ -f "$bootstrap_rehearsal_complete" \
      && ! -f "$bootstrap_rehearsal_dir/needs_attention" \
      && -f "$bootstrap_rehearsal_dir/comparison.json" ]]; then
  candidate_args+=(--candidate "bootstrap=$bootstrap_rehearsal_dir/comparison.json")
elif [[ -f "$bootstrap_rehearsal_early_stopped" ]]; then
  log "bootstrap symmetric-tail rehearsal was scientifically early-stopped; excluding cancelled-gradient objective"
else
  log "bootstrap causal rehearsal unavailable; selecting from completed families"
fi
if [[ -f "$bootstrap_directional_complete" \
      && ! -f "$bootstrap_directional_dir/needs_attention" \
      && -f "$bootstrap_directional_dir/comparison.json" ]]; then
  candidate_args+=(--candidate "bootstrap_directional=$bootstrap_directional_dir/comparison.json")
else
  log "bootstrap directional-tail rehearsal unavailable; selecting from completed families"
fi
if [[ -f "$donor_directional_complete" \
      && ! -f "$donor_directional_dir/needs_attention" \
      && -f "$donor_directional_dir/comparison.json" ]]; then
  candidate_args+=(--candidate "donor_directional=$donor_directional_dir/comparison.json")
else
  log "persistent donor-directional rehearsal unavailable; selecting from completed families"
fi
if [[ -f "$cf4_complete" && ! -f "$cf4_attention" \
      && -f "$cf4_dir/comparison.json" ]]; then
  candidate_args+=(--candidate "cf4=$cf4_dir/comparison.json")
else
  log "CF4 multi-query diagnostic unavailable; selecting from completed families"
fi
python scripts/select_m4_causal_candidate.py \
  "${candidate_args[@]}" \
  --output "$selection_json" >> "$state_dir/controller.log" 2>&1 \
  || fail "could not compare sparse and dense causal diagnostics"
if ! python - "$selection_json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
raise SystemExit(0 if x.get('scientific_gate_pass') is True else 1)
PY
then
  log "sparse and dense causal diagnostics missed C4; starting strength escalation"
  # Another controller may have started escalation speculatively while the dense
  # diagnostics were queued.  The escalation watcher uses a non-blocking flock,
  # so a second invocation can legitimately return 0 before the active watcher
  # has published its atomic completion marker.  Keep waiting for that owner (or
  # restart the watcher if its PID disappears) instead of misclassifying the
  # lock hand-off as a completed escalation.
  while [[ ! -f "$escalation_dir/complete" ]]; do
    heartbeat
    [[ ! -f "$escalation_dir/needs_attention" ]] \
      || fail "causal-strength escalation controller failed"
    if [[ -f "$escalation_dir/controller.active.pid" ]] \
        && kill -0 "$(cat "$escalation_dir/controller.active.pid")" 2>/dev/null; then
      sleep 120
      continue
    fi
    bash scripts/watch_m4_causal_strength_escalation.sh \
      || fail "causal-strength escalation controller failed"
  done
  [[ -f "$escalation_dir/complete" && ! -f "$escalation_dir/needs_attention" ]] \
    || fail "causal-strength escalation did not complete cleanly"
  python scripts/select_m4_causal_candidate.py \
    "${candidate_args[@]}" \
    --candidate "escalation=$escalation_dir/comparison.json" \
    --output "$selection_json" >> "$state_dir/controller.log" 2>&1 \
    || fail "could not select from escalated causal diagnostics"
fi

readarray -t ranked_candidates < <(python - "$selection_json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
assert x.get('stage') == 'M4_causal_candidate_selection', x
assert x.get('scientific_gate_pass') is True, x
allowed_branches = {
    'raw4', 'mean_normalized', 'raw8', 'mean2_normalized',
    'repeat100_w1', 'repeat100_w4'
    , 'repeat100_sum_w1', 'repeat100_sum_w4'
    , 'bootstrap_sum_w1', 'bootstrap_sum_w4'
    , 'bootstrap_native25_sum_w1', 'bootstrap_native25_sum_w4'
    , 'bootstrap_balanced_sum_w1', 'bootstrap_balanced_sum_w4'
    , 'bootstrap_native25_balanced_sum_w1', 'bootstrap_native25_balanced_sum_w4'
    , 'donor_directional_sum_w1', 'donor_directional_sum_w4'
    , 'cf4_bootstrap_symmetric_w1', 'cf4_donor_directional_w1'
}
allowed_families = {
    'sparse', 'dense', 'sum_aligned', 'bootstrap',
    'bootstrap_directional', 'escalation',
    'donor_directional',
    'cf4',
}
ranked = x.get('ranked_candidates') or []
assert ranked, x
for row in ranked:
    branch = row.get('branch')
    family = row.get('family')
    assert row.get('eligible') is True, row
    assert branch in allowed_branches, row
    assert family in allowed_families, row
    print(f'{family}\t{branch}')
PY
) || fail "no C4-qualified causal-strength branch after escalation"

selected_candidate=false
for candidate in "${ranked_candidates[@]}"; do
  IFS=$'\t' read -r family branch <<< "$candidate"
  full_job="m4-causal-rescue-full-${branch//_/-}-seed42"
  stand_down_if_original_passed
  if [[ "$family" == dense ]]; then
    full_base="$project_dir/amlt/feeds_m4_causal_rehearsal_20260731.yaml"
  elif [[ "$family" == sum_aligned ]]; then
    full_base="$project_dir/amlt/feeds_m4_causal_rehearsal_sum_20260731.yaml"
  elif [[ "$family" == bootstrap ]]; then
    full_base="$project_dir/amlt/feeds_m4_causal_bootstrap_rehearsal_selected_20260731.yaml"
  elif [[ "$family" == bootstrap_directional ]]; then
    full_base="$project_dir/amlt/feeds_m4_causal_bootstrap_directional_tail_20260731.yaml"
  elif [[ "$family" == donor_directional ]]; then
    full_base="$project_dir/amlt/feeds_m4_causal_donor_directional_20260731.yaml"
  elif [[ "$family" == cf4 ]]; then
    full_base="$project_dir/amlt/feeds_m4_cf4_diagnostic_20260731.yaml"
  elif [[ "$family" == escalation ]]; then
    full_base="$project_dir/amlt/feeds_m4_causal_strength_escalation_20260731.yaml"
  else
    full_base="$project_dir/amlt/feeds_m4_causal_strength_branches_20260731.yaml"
  fi
  candidate_dir="$state_dir/candidates/$branch"
  mkdir -p "$candidate_dir"
  log "trying ranked causal-rescue family=$family branch=$branch"

  python scripts/render_m4_causal_rescue.py --mode full --base "$full_base" \
    --branches "$branch" --output "$full_config" >> "$state_dir/controller.log" 2>&1 \
    || fail "could not render causal-rescue full config"
  python scripts/validate_m4_causal_rescue_amlt_config.py --config "$full_config" \
    --jobs "$full_job" >> "$state_dir/controller.log" 2>&1 \
    || fail "rendered causal-rescue config failed validation"

  full_base_experiment="recurrent-m4-causal-rescue-full-${branch//_/-}-seed42-20260731"
  full_passed=false
  for recovery_wave in 0 1 2 3; do
    full_experiment="$full_base_experiment"
    (( recovery_wave == 0 )) || full_experiment="${full_base_experiment}-recovery${recovery_wave}"
    wait_quota 4 "rescue_${branch}_full_recovery${recovery_wave}"
    submit_config "$full_config" "$full_experiment" \
      "C4-qualified M4 causal-rescue branch=$branch, exact one epoch with atomic recovery" \
      rescue "$full_job"
    if wait_pass_recoverable "$full_experiment" 1 "causal-rescue one-epoch M4"; then
      full_passed=true
      break
    fi
  done
  if [[ "$full_passed" != true ]]; then
    log "candidate infrastructure failed after recovery waves: family=$family branch=$branch"
    touch "$candidate_dir/infrastructure_failed"
    continue
  fi

  full_download="$candidate_dir/full_download/$full_experiment"
  mkdir -p "$full_download"
  amlt results download "$full_experiment" --status pass \
    --include 'adapter_model.safetensors' --include 'adapter_config.json' \
    --include 'recurrent_state.pt' --include 'completion.json' \
    --include 'validation_metrics.json' --include 'causal_validation_metrics.json' \
    --include 'training_metrics.json' --include 'run_config.json' \
    --include 'gradient_audit.json' --include 'full_training_summary.json' \
    --include 'recurrent_causal_gate.json' --output "$full_download" \
    >> "$state_dir/controller.log" 2>&1 \
    || fail "could not download causal-rescue artifact"
  if ! python scripts/validate_m4_causal_rescue_result.py --root "$full_download" \
      --branch "$branch" --output "$candidate_dir/full_gate.json" \
      >> "$state_dir/controller.log" 2>&1; then
    log "candidate scientific gate failed after exact one epoch: family=$family branch=$branch"
    touch "$candidate_dir/scientific_gate_failed"
    continue
  fi

  python scripts/render_m4_causal_rescue.py --mode eval --base "$eval_base" \
    --output "$eval_config" >> "$state_dir/controller.log" 2>&1 \
    || fail "could not render causal-rescue evaluation config"
  python scripts/validate_m4_eval_amlt_config.py --config "$eval_config" \
    >> "$state_dir/controller.log" 2>&1 \
    || fail "rendered causal-rescue evaluation failed validation"
  eval_jobs=(m4-eval-t1-t6-seed42 m4-eval-t2-seed42 m4-eval-t3-seed42 m4-eval-t5-seed42 m4-eval-c4-seed42)
  eval_base_experiment="recurrent-m4-causal-rescue-formal-eval-${branch//_/-}-seed42-20260731"
  eval_passed=false
  for recovery_wave in 0 1 2 3; do
    eval_experiment="$eval_base_experiment"
    (( recovery_wave == 0 )) || eval_experiment="${eval_base_experiment}-recovery${recovery_wave}"
    wait_quota 5 "rescue_${branch}_eval_recovery${recovery_wave}"
    submit_config "$eval_config" "$eval_experiment" \
      "Formal T1 T2 T3 T5 T6 and frozen C4 for C4-qualified causal-rescue M4" \
      eval "${eval_jobs[@]}"
    if wait_pass_recoverable "$eval_experiment" 5 "causal-rescue formal evaluation"; then
      eval_passed=true
      break
    fi
  done
  if [[ "$eval_passed" != true ]]; then
    log "candidate formal-eval infrastructure failed after recovery waves: family=$family branch=$branch"
    touch "$candidate_dir/eval_infrastructure_failed"
    continue
  fi

  eval_download="$candidate_dir/eval_download/$eval_experiment"
  mkdir -p "$eval_download"
  amlt results download "$eval_experiment" --status pass \
    --include 'metrics.json' --include 'recurrent_causal_gate.json' \
    --include 'predictions.jsonl' --include 'predictions_redacted.jsonl' \
    --include 'manifest_audit.json' --include 'selection.txt' \
    --output "$eval_download" >> "$state_dir/controller.log" 2>&1 \
    || fail "could not download causal-rescue formal evaluation"
  if ! python scripts/validate_m4_pilot_result.py --root "$eval_download" \
      --output "$candidate_dir/pilot_scientific_gate.json" \
      >> "$state_dir/controller.log" 2>&1; then
    log "candidate scientific gate failed in formal task/C4 evaluation: family=$family branch=$branch"
    touch "$candidate_dir/formal_scientific_gate_failed"
    continue
  fi

  cp "$candidate_dir/full_gate.json" "$state_dir/full_gate.json"
  cp "$candidate_dir/pilot_scientific_gate.json" "$state_dir/pilot_scientific_gate.json"
  printf '%s\n' "$branch" > "$state_dir/selected_branch.txt"
  printf '%s\n' "$family" > "$state_dir/selected_family.txt"
  selected_candidate=true
  log "selected_candidate=true family=$family branch=$branch"
  break
done
[[ "$selected_candidate" == true ]] \
  || fail "all eligible causal-rescue candidates failed exact-epoch or formal scientific gates"

az storage blob download --account-name "$account" --container-name "$container" \
  --auth-mode login --name "$isolated_receipt" \
  --file "$state_dir/selected_receipt.json" --overwrite \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "could not download isolated causal-rescue receipt"
python - "$state_dir/selected_receipt.json" "$branch" <<'PY' \
  || fail "isolated causal-rescue receipt is invalid"
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
assert x.get('status') == 'candidate_complete', x
assert x.get('causal_rescue_branch') == sys.argv[2], x
assert float(x.get('C4_D_specific', -1)) >= 0.15, x
assert x.get('mask_protocol') == 'answer_anchor_visible_v2', x
PY
az storage blob download --account-name "$account" --container-name "$container" \
  --auth-mode login --name "$main_receipt" \
  --file "$state_dir/pre_rescue_selected_receipt.json" --overwrite \
  >> "$state_dir/controller.log" 2>&1 || true
az storage blob upload --account-name "$account" --container-name "$container" \
  --auth-mode login --name "$main_receipt" \
  --file "$state_dir/selected_receipt.json" --overwrite \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "could not promote causal-rescue receipt"
az storage blob download --account-name "$account" --container-name "$container" \
  --auth-mode login --name "$main_receipt" \
  --file "$state_dir/promoted_receipt_verification.json" --overwrite \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "could not verify promoted causal-rescue receipt"
cmp -s "$state_dir/selected_receipt.json" "$state_dir/promoted_receipt_verification.json" \
  || fail "promoted causal-rescue receipt differs from selected receipt"

cp "$state_dir/pilot_scientific_gate.json" "$m4_dir/pilot_scientific_gate.json"
rm -f "$m4_dir/needs_attention" "$m4_dir/causal_rescue_in_progress"
touch "$m4_dir/pilot_complete" "$state_dir/complete"
log "causal-rescue M4 passed formal gates and was promoted into the unified-model pipeline"
