#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
parent="$project/results/state8_exact_diagnosis_pipeline_20260808"
state="$project/results/state8_exact_consolidation_milestones_20260808"
config="$project/amlt/feeds_state8_exact_checkpoint200_eval_20260808.yaml"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
steps=(200 400 600 800)
max_recovery_waves=24
rescue_state="$project/results/state8_exact_scientific_rescue_20260808"

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

while [[ ! -s "$parent/selection.json" ]]; do
  heartbeat
  [[ -f "$parent/needs_attention" ]] && fail "parent exact pipeline needs attention"
  sleep_unlocked 60
done

if "$python_bin" - "$parent/selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_exact_diagnosis_curriculum_joint_promotion_v1',x
raise SystemExit(0 if x['pass'] is True and x['selected']['qualified'] is True else 1)
PY
then
  touch "$state/not_required" "$state/complete"
  heartbeat
  log "exact final checkpoint passed; conditional milestone curve not required"
  exit 0
fi

"$python_bin" - "$parent/selection.json" <<'PY' \
  || fail "parent exact selection is not a valid failed candidate"
import json,sys
x=json.load(open(sys.argv[1]))
assert x['stage']=='state8_exact_diagnosis_curriculum_joint_promotion_v1',x
assert x['pass'] is False and x.get('selected') is None,x
assert len(x.get('candidates') or [])==1 and x['candidates'][0]['qualified'] is False,x
PY

"$python_bin" scripts/validate_state8_exact_consolidation_milestones_amlt.py \
  --config "$config" >> "$state/controller.log" 2>&1 \
  || fail "milestone AMLT source validation failed"

jobs=()
for step in "${steps[@]}"; do jobs+=("state8-exact-consolidation-step${step}-eval"); done
printf '%s\n' "${jobs[@]}" > "$state/required_jobs.txt"

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
  local requested=("$@") selectors=() rc status phase audit available auditable
  local reserve capacity
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
  # If this watcher wins the shared lock immediately after the parent fails,
  # leave room for the three higher-priority scientific-rescue arms. Once
  # those jobs are submitted they are already reflected in the active audit.
  reserve=0
  if [[ ! -f "$rescue_state/complete" ]] \
      && ! compgen -G "$rescue_state/submitted.*" > /dev/null; then
    reserve=3
  fi
  capacity=$(( available - reserve ))
  (( capacity > ${#requested[@]} )) && capacity=${#requested[@]}
  (( capacity > 0 )) || { flock -u 8; return 1; }
  requested=("${requested[@]:0:capacity}")
  for job in "${requested[@]}"; do selectors+=(":$job"); done
  printf '%s\n' "${requested[@]}" > "$state/$experiment.submitted_jobs.txt"
  "$python_bin" scripts/validate_state8_exact_consolidation_milestones_amlt.py \
    --config "$config" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_exact_consolidation_milestones_amlt.py \
    --config "$state/$experiment.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Conditional read-only exact State8 diagnosis-first consolidation milestones" \
    > "$state/$experiment.submission.txt" 2>&1
  rc=$?
  if (( rc != 0 )); then
    status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
      || { flock -u 8; return "$rc"; }
    phase="$(status_phase "$status")"
    [[ "$phase" == active || "$phase" == pass ]] || { flock -u 8; return "$rc"; }
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
 if str(row.get('status') or '').casefold()=='pass': print(str(row['job_name']).lstrip(':'))
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

experiment=state8-exact-consolidation-milestones-20260808
wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260808 ]] && wave="${BASH_REMATCH[1]}"
fi
while [[ ! -f "$state/remote_passed" ]]; do
  missing=()
  for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  (( ${#missing[@]} > 0 )) || { touch "$state/remote_passed"; break; }
  if [[ ! -f "$state/submitted.$experiment" ]]; then
    until submit_jobs "$experiment" "${missing[@]}"; do
      heartbeat; log "transient milestone submission failure; retrying"; sleep_unlocked 120
    done
    printf '%s\n' "$experiment" > "$state/active_experiment.txt"
    touch "$state/submitted.$experiment"
    log "submitted milestone experiment=$experiment jobs=${missing[*]}"
  fi
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" "$status" || fail "could not preserve passed milestones"
  missing=()
  for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  (( ${#missing[@]} > 0 )) || { touch "$state/remote_passed"; break; }
  [[ "$(status_phase "$status")" == active ]] && { sleep_unlocked 120; continue; }
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" \
    > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) || fail "milestones exhausted recovery waves"
  wave=$((wave+1)); experiment="state8-exact-consolidation-milestones-recovery${wave}-20260808"
  log "recovering failed milestone jobs wave=$wave jobs=${missing[*]}"
done

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
for index in "${!steps[@]}"; do
  step="${steps[$index]}"; job="${jobs[$index]}"; root="$state/step$step"
  mkdir -p "$root/download" "$root/t3_semantic" "$root/c4_semantic"
  mapfile -t job_roots < <(find "$state/preserved_results" -type d -name "$job")
  [[ ${#job_roots[@]} -eq 1 ]] || fail "milestone artifact root drift step=$step"
  if [[ ! -f "$root/artifacts_materialized" ]]; then
    cp -a "${job_roots[0]}" "$root/download/"
    touch "$root/artifacts_materialized"
  fi
  artifact="$root/download/$job"
  formal="$artifact/eval/formal_t3_300/predictions.jsonl"
  diag_pred="$artifact/eval/diagnosis_c4_50/judge_predictions.jsonl"
  diag_eval="$artifact/eval/diagnosis_c4_50/judge_eval.jsonl"
  diag_metrics="$artifact/eval/diagnosis_c4_50/metrics.json"
  for path in "$formal" "$diag_pred" "$diag_eval" "$diag_metrics"; do
    [[ -s "$path" ]] || fail "missing milestone evaluation artifact step=$step path=$path"
  done
  if [[ ! -f "$root/t3_semantic_complete" ]]; then
    PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$formal" --eval-jsonl "$eval_jsonl" \
      --output-dir "$root/t3_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$root/t3_semantic/cache.jsonl" --judge-audit-samples 100 \
      --seed 42 >> "$root/t3_semantic/judge.log" 2>&1 \
      || fail "milestone T3 semantic judge failed step=$step"
    touch "$root/t3_semantic_complete"
  fi
  if [[ ! -f "$root/c4_semantic_complete" ]]; then
    PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$diag_pred" --eval-jsonl "$diag_eval" \
      --output-dir "$root/c4_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$root/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
      --seed 42 >> "$root/c4_semantic/judge.log" 2>&1 \
      || fail "milestone diagnosis-C4 judge failed step=$step"
    "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
      --judge-redacted "$root/c4_semantic/predictions_redacted.jsonl" \
      --generation-metrics "$diag_metrics" --output "$root/c4_semantic/summary.json" \
      >> "$root/c4_semantic/summarize.log" 2>&1 \
      || fail "milestone diagnosis-C4 summary failed step=$step"
    touch "$root/c4_semantic_complete"
  fi
done

set +e
"$python_bin" scripts/select_state8_exact_consolidation_milestones.py \
  --root "$state" --output "$state/selection.json" > "$state/selection.log" 2>&1
selection_rc=$?
set -e
if (( selection_rc == 0 )); then touch "$state/diagnostic_gate_pass"; fi
if (( selection_rc == 2 )); then touch "$state/diagnostic_gate_failed"; fi
(( selection_rc == 0 || selection_rc == 2 )) || fail "milestone selector crashed rc=$selection_rc"
touch "$state/complete"; heartbeat
log "conditional exact consolidation milestone curve complete"
