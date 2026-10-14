#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
parent="$project/results/state8_exact_diagnosis_pipeline_20260808"
state="$project/results/state8_exact_scientific_rescue_20260808"
source_config="$project/amlt/feeds_state8_exact_scientific_rescue_20260808.yaml"
config="$state/materialized.yaml"
bundle="$project/state8_exact_scientific_rescue_bundle_20260808"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
max_recovery_waves=24

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi
mkdir -p "$state/preserved_results"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
sleep_unlocked(){ sleep "$1" 9>&-; }
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

selection_passes(){
  "$python_bin" - "$1" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_exact_diagnosis_curriculum_joint_promotion_v1',x
raise SystemExit(0 if x['pass'] is True and x['selected']['qualified'] is True else 1)
PY
}

selection_fails(){
  "$python_bin" - "$1" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_exact_diagnosis_curriculum_joint_promotion_v1',x
assert x['pass'] is False and x.get('selected') is None,x
assert len(x.get('candidates') or [])==1 and x['candidates'][0]['qualified'] is False,x
PY
}

while [[ ! -s "$parent/selection.json" ]]; do
  heartbeat
  [[ -f "$parent/needs_attention" ]] && fail "parent exact pipeline needs attention before scientific selection"
  sleep_unlocked 60
done

if selection_passes "$parent/selection.json"; then
  touch "$state/not_required" "$state/complete"
  heartbeat
  log "exact diagnosis curriculum passed; conditional scientific rescue not required"
  exit 0
fi
selection_fails "$parent/selection.json" \
  || fail "parent exact selection is neither a valid pass nor failed candidate"

if [[ ! -s "$config" ]]; then
  "$python_bin" scripts/materialize_state8_exact_scientific_rescue_amlt.py \
    --source "$source_config" --selection "$parent/selection.json" \
    --output "$config" >> "$state/controller.log" 2>&1 \
    || fail "conditional rescue materialization failed"
fi
"$python_bin" scripts/validate_state8_exact_scientific_rescue_amlt.py \
  --config "$config" --selection "$parent/selection.json" --bundle "$bundle" \
  >> "$state/controller.log" 2>&1 \
  || fail "conditional rescue source validation failed"

mapfile -t jobs < <("$python_bin" - "$config" <<'PY'
import sys,yaml
for row in yaml.safe_load(open(sys.argv[1]))['jobs']:
 print(row['name'])
PY
)
mapfile -t modes < <("$python_bin" - "$config" <<'PY'
import sys,yaml
for row in yaml.safe_load(open(sys.argv[1]))['jobs']:
 print(row['submit_args']['env']['STATE8_RESCUE_MODE'])
PY
)
(( ${#jobs[@]} >= 1 && ${#jobs[@]} == ${#modes[@]} )) \
  || fail "empty or mismatched conditional rescue jobs"
printf '%s\n' "${jobs[@]}" > "$state/required_jobs.txt"
printf '%s\n' "${modes[@]}" > "$state/required_modes.txt"

status_phase(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
states={str(row.get('status') or '').casefold() for row in rows}
active={'preparing','pre-queued','queued','running','paused','resuming'}
if states and states <= {'pass'}: print('pass')
elif states & active: print('active')
elif states: print('terminal')
else: print('invalid')
PY
}

submit_jobs(){
  local experiment="$1"; shift
  local requested=("$@") selectors=() rc status phase audit available capacity auditable
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  audit="$state/$experiment.active_a100.json"
  "$python_bin" scripts/count_active_a100_requests.py \
    --user-limit 32 --recent 500 --output "$audit" \
    > /dev/null 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  read -r available auditable < <("$python_bin" - "$audit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(int(x['available_user_slots']), int(bool(x['auditable'])))
PY
  )
  [[ "$auditable" == 1 ]] || { flock -u 8; return 1; }
  capacity=$(( available < ${#requested[@]} ? available : ${#requested[@]} ))
  (( capacity > 0 )) || { flock -u 8; return 1; }
  requested=("${requested[@]:0:capacity}")
  for job in "${requested[@]}"; do selectors+=(":$job"); done
  printf '%s\n' "${requested[@]}" > "$state/$experiment.submitted_jobs.txt"
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" \
    2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_exact_scientific_rescue_amlt.py \
    --config "$state/$experiment.preflight.yaml" \
    --selection "$parent/selection.json" --bundle "$bundle" \
    >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds \
    > "$state/$experiment.quota.json" 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Conditional exact State8 scientific rescue: hash-bound semantic-preserve and/or causal-taskanchor" \
    > "$state/$experiment.submission.txt" 2>&1
  rc=$?
  if (( rc != 0 )); then
    status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
      || { flock -u 8; return "$rc"; }
    phase="$(status_phase "$status")"
    [[ "$phase" == active || "$phase" == pass ]] \
      || { flock -u 8; return "$rc"; }
    printf '\nsubmission_rc=%s but exact experiment exists phase=%s\n' \
      "$rc" "$phase" >> "$state/$experiment.submission.txt"
  fi
  flock -u 8
}

preserve_passed(){
  local experiment="$1" status="$2" job
  mapfile -t observed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass':
  print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${observed[@]}"; do
    [[ -f "$state/passed.$job" ]] && continue
    amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 \
      || return 1
    touch "$state/passed.$job"
  done
}

experiment=state8-exact-scientific-rescue-20260808
wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260808 ]] \
    && wave="${BASH_REMATCH[1]}"
fi
while [[ ! -f "$state/remote_passed" ]]; do
  missing=()
  for job in "${jobs[@]}"; do
    [[ -f "$state/passed.$job" ]] || missing+=("$job")
  done
  if (( ${#missing[@]} == 0 )); then
    touch "$state/remote_passed"
    break
  fi
  if [[ ! -f "$state/submitted.$experiment" ]]; then
    until submit_jobs "$experiment" "${missing[@]}"; do
      heartbeat
      log "transient rescue submission failure experiment=$experiment; retrying"
      sleep_unlocked 120
    done
    printf '%s\n' "$experiment" > "$state/active_experiment.txt"
    printf '%s\n' "${missing[@]}" > "$state/active_jobs.txt"
    touch "$state/submitted.$experiment"
    log "submitted conditional exact rescue experiment=$experiment jobs=${missing[*]}"
  fi
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" "$status" \
    || fail "could not preserve passed rescue jobs"
  missing=()
  for job in "${jobs[@]}"; do
    [[ -f "$state/passed.$job" ]] || missing+=("$job")
  done
  if (( ${#missing[@]} == 0 )); then
    touch "$state/remote_passed"
    break
  fi
  phase="$(status_phase "$status")"
  [[ "$phase" == active ]] && { sleep_unlocked 120; continue; }
  timeout 90 amlt show "$experiment" \
    > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" \
    > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) \
    || fail "conditional exact rescue exhausted recovery waves"
  wave=$((wave+1))
  experiment="state8-exact-scientific-rescue-recovery${wave}-20260808"
  log "recovering failed rescue jobs wave=$wave jobs=${missing[*]}"
done

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
for index in "${!jobs[@]}"; do
  job="${jobs[$index]}"
  mode="${modes[$index]}"
  root="$state/$mode"
  mkdir -p "$root/download" "$root/t3_semantic" "$root/c4_semantic"
  mapfile -t job_roots < <(find "$state/preserved_results" -type d -name "$job")
  [[ ${#job_roots[@]} -eq 1 ]] \
    || fail "rescue job artifact root cardinality drift job=$job"
  if [[ ! -f "$root/artifacts_materialized" ]]; then
    cp -a "${job_roots[0]}" "$root/download/"
    touch "$root/artifacts_materialized"
  fi
  mapfile -t formal < <(find "$root/download" -type f \
    -path "*/$job/eval/formal_t3_300/predictions.jsonl")
  mapfile -t diag_pred < <(find "$root/download" -type f \
    -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
  mapfile -t diag_eval < <(find "$root/download" -type f \
    -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
  mapfile -t diag_metrics < <(find "$root/download" -type f \
    -path "*/$job/eval/diagnosis_c4_50/metrics.json")
  [[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 \
      && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
    || fail "rescue evaluation artifact cardinality drift job=$job"
  if [[ ! -f "$root/t3_semantic_complete" ]]; then
    attempt=0
    until PYTHONPATH="$judge_path" "$judge_python" \
        scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" \
        --output-dir "$root/t3_semantic" --judge-backend trapi \
        --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$root/t3_semantic/cache.jsonl" \
        --judge-audit-samples 100 --seed 42 \
        >> "$root/t3_semantic/judge.log" 2>&1; do
      attempt=$((attempt+1))
      (( attempt < 30 )) || fail "rescue T3 judge exhausted job=$job"
      heartbeat
      sleep_unlocked 120
    done
    touch "$root/t3_semantic_complete"
  fi
  if [[ ! -f "$root/c4_semantic_complete" ]]; then
    attempt=0
    until PYTHONPATH="$judge_path" "$judge_python" \
        scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
        --output-dir "$root/c4_semantic" --judge-backend trapi \
        --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$root/c4_semantic/cache.jsonl" \
        --judge-audit-samples 100 --seed 42 \
        >> "$root/c4_semantic/judge.log" 2>&1; do
      attempt=$((attempt+1))
      (( attempt < 30 )) || fail "rescue diagnosis C4 judge exhausted job=$job"
      heartbeat
      sleep_unlocked 120
    done
    "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
      --judge-redacted "$root/c4_semantic/predictions_redacted.jsonl" \
      --generation-metrics "${diag_metrics[0]}" \
      --output "$root/c4_semantic/summary.json" \
      >> "$root/c4_semantic/summarize.log" 2>&1 \
      || fail "rescue diagnosis C4 summary failed job=$job"
    touch "$root/c4_semantic_complete"
  fi
done

selector_args=()
for mode in "${modes[@]}"; do selector_args+=(--mode "$mode"); done
set +e
"$python_bin" scripts/select_state8_exact_scientific_rescue.py \
  --root "$state" "${selector_args[@]}" --output "$state/selection.json" \
  > "$state/selection.log" 2>&1
selection_rc=$?
set -e
if (( selection_rc == 0 )); then
  az storage blob upload --auth-mode login \
    --account-name csnewsandfeeds4150361735 --container-name amulet \
    --name studentmodeling/gates/state8_exact_scientific_rescue_v1/selection.json \
    --file "$state/selection.json" --overwrite \
    > "$state/promotion_upload.txt" 2>&1 \
    || fail "exact rescue promotion gate upload failed"
  touch "$state/scientific_gate_pass"
elif (( selection_rc == 2 )); then
  touch "$state/scientific_gate_failed"
else
  fail "exact rescue scientific selector crashed rc=$selection_rc"
fi
touch "$state/complete"
heartbeat
log "conditional exact scientific rescue reached terminal joint selection"
