#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
bridge="$project/results/state8_strong1_widen8_bridge_20260808"
fallback="$bridge/single_fallback_monitor"
midpoint="$project/results/state8_midpoint_diagnosis_first_20260808"
exact_diagnosis="$project/results/state8_exact_diagnosis_pipeline_20260808"
state="$bridge/complete_pipeline/rescue"
bundle="$project/strong1_rescue_bundle_20260808"
config="$project/amlt/feeds_state8_strong1_rescue_generated_20260808.yaml"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
jobs=(state8-strong1-rescue-taxonomy_joint-seed61 state8-strong1-rescue-taskanchor_c4strong-seed62)
slugs=(taxonomy_joint taskanchor_c4strong)
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
# Do not let long sleeps inherit the controller lock.  A killed supervisor can
# otherwise leave an orphaned sleep holding fd 9 and silently prevent its
# wrapper from restarting the controller.
sleep_unlocked(){ sleep "$1" 9>&-; }
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

selection_passes(){
  "$python_bin" - "$1" <<'PY'
import json,sys
try: x=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
assert x['stage'] in {
 'state8_strong1_widen8_joint_promotion_v1',
 'state8_midpoint_diagnosis_first_joint_promotion_v1',
 'state8_exact_diagnosis_curriculum_joint_promotion_v1',
},x
assert x['pass'] is True and x['selected']['qualified'] is True,x
PY
}

selection_fails(){
  "$python_bin" - "$1" <<'PY'
import json,sys
try: x=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
assert x['stage']=='state8_strong1_widen8_joint_promotion_v1',x
assert x['pass'] is False and x.get('selected') is None,x
candidates=x.get('candidates')
assert isinstance(candidates,list) and candidates,candidates
assert all(row.get('qualified') is False for row in candidates),candidates
PY
}

# Rescue is a conditional branch.  It must never compete with a qualified
# primary bridge.  Start as soon as any fully evaluated primary candidate has
# failed the joint gate: waiting for an independent sibling that may still be
# queued for hours wastes the completed candidate's diagnosis-C4 evidence.
# A later sibling remains free to train and can still win the central selector.
while true; do
  heartbeat
  for path in "$bridge/selection.json" "$fallback/selection.json" \
      "$midpoint/selection.json" "$exact_diagnosis/selection.json"; do
    if [[ -s "$path" ]] && selection_passes "$path"; then
      touch "$state/not_required" "$state/complete"
      log "qualified primary bridge exists; rescue not required"
      exit 0
    fi
  done
  failed_selection_available=false
  for path in "$bridge/selection.json" "$fallback/selection.json"; do
    [[ -s "$path" ]] || continue
    if selection_fails "$path"; then
      failed_selection_available=true
    fi
  done
  [[ "$failed_selection_available" == true ]] && break
  # The legacy bridge arms can be deliberately retired before evaluation when
  # the diagnosis-first full curriculum supersedes them.  In that case they
  # cannot produce the failed-selection artifact this older rescue controller
  # was designed to consume.  Treat this rescue as not required so the central
  # controller can wait on the hash-bound exact diagnosis branch instead of
  # deadlocking forever on impossible legacy inputs.
  if [[ ( -f "$bridge/superseded" || -f "$bridge/superseded_before_run" ) \
        && ( -f "$fallback/superseded" \
             || -f "$fallback/retired_for_diagnosis_first_priority" \
             || -f "$fallback/superseded_before_run" ) \
        && -s "$exact_diagnosis/active_experiment.txt" ]]; then
    touch "$state/not_required" "$state/complete"
    log "legacy rescue sources were retired for exact diagnosis-first curriculum; rescue not required"
    exit 0
  fi
  sleep_unlocked 120
done

selection_args=()
for path in "$bridge/selection.json" "$fallback/selection.json"; do
  [[ -s "$path" ]] && selection_args+=(--selection "$path")
done
(( ${#selection_args[@]} > 0 )) || fail "failed primary bridge has no selection artifacts"
"$python_bin" scripts/select_state8_strong1_rescue_source.py \
  "${selection_args[@]}" --output "$state/source_selection.json" \
  > "$state/source_selection.log" 2>&1 \
  || fail "rescue source selection failed"
"$python_bin" scripts/materialize_state8_strong1_rescue_amlt.py \
  --selection "$state/source_selection.json" --output "$config" \
  >> "$state/controller.log" 2>&1 || fail "rescue config materialization failed"
"$python_bin" scripts/validate_state8_strong1_rescue_amlt.py \
  --config "$config" --selection "$state/source_selection.json" --bundle "$bundle" \
  >> "$state/controller.log" 2>&1 || fail "rescue config validation failed"

status_json(){ timeout 90 amlt --json-tables status "$1" --hide-urls 2>/dev/null; }
status_class(){
  STATUS="$1" EXPECTED="$2" "$python_bin" - <<'PY'
import json,os
try: rows=json.loads(os.environ['STATUS'])
except Exception: print('unknown'); raise SystemExit
if len(rows)!=int(os.environ['EXPECTED']): print('unknown'); raise SystemExit
states={str(x.get('status') or '').casefold() for x in rows}
active={'preparing','pre-queued','queued','running','paused','resuming'}
terminal={'failed','fail','cancelled','canceled','stopped','killed'}
print('active' if states & active else ('pass' if states=={'pass'} else ('failed' if states & terminal else 'unknown')))
PY
}
preserve_passed(){
  local experiment="$1"
  local status pass_count
  status="$(status_json "$experiment")" || return 1
  pass_count="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
print(sum(str(x.get('status')).casefold()=='pass' for x in json.loads(os.environ['STATUS'])))
PY
)" || return 1
  (( pass_count > 0 )) || return 0
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/preserved_results" >> "$state/controller.log" 2>&1
}
submit_jobs(){
  local experiment="$1"; shift; local names=("$@")
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" "${names[@]/#/:}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_strong1_rescue_amlt.py \
    --config "$state/$experiment.preflight.yaml" \
    --selection "$state/source_selection.json" --bundle "$bundle" --jobs "${names[@]}" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  amlt run "$config" "${names[@]/#/:}" "$experiment" -t Feeds -y \
    --description "Conditional strict-state-only strong1 rescue; source and artifacts hash-bound; experiment=$experiment" \
    > "$state/$experiment.submission.txt" 2>&1
  rc=$?; flock -u 8; return "$rc"
}

experiment=state8-strong1-rescue-20260808; wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260808 ]] && wave="${BASH_REMATCH[1]}"
fi
if [[ ! -f "$state/submitted.$experiment" ]]; then
  submit_jobs "$experiment" "${jobs[@]}" || fail "initial rescue submission failed"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${jobs[@]}" > "$state/active_jobs.txt"
  touch "$state/submitted.$experiment"
  log "submitted conditional rescue experiment=$experiment"
fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(status_json "$experiment")" || { sleep_unlocked 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  expected="$(wc -l < "$state/active_jobs.txt")"
  class="$(status_class "$status" "$expected")"
  [[ "$class" == active || "$class" == unknown ]] && { sleep_unlocked 120; continue; }
  preserve_passed "$experiment" || fail "could not preserve rescue siblings"
  mapfile -t passed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status')).casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${passed[@]}"; do touch "$state/passed.$job"; done
  if [[ "$class" == pass ]]; then
    all=1; for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || all=0; done
    (( all )) || fail "rescue pass accounting lost an arm"
    touch "$state/remote_passed"; break
  fi
  mapfile -t failed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
terminal={'failed','fail','cancelled','canceled','stopped','killed'}
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status')).casefold() in terminal: print(str(row['job_name']).lstrip(':'))
PY
  )
  (( ${#failed[@]} > 0 && wave < max_recovery_waves )) \
    || fail "rescue terminal state has no recoverable arms"
  wave=$((wave+1)); experiment="state8-strong1-rescue-recovery${wave}-20260808"
  submit_jobs "$experiment" "${failed[@]}" || fail "rescue recovery submission failed wave=$wave"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${failed[@]}" > "$state/active_jobs.txt"
  log "submitted rescue recovery wave=$wave jobs=${failed[*]}"
done

rm -rf "$state/download"; mkdir -p "$state/download"
cp -a "$state/preserved_results/." "$state/download/" || fail "rescue result materialization failed"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
for index in "${!jobs[@]}"; do
  job="${jobs[$index]}"; slug="${slugs[$index]}"; root="$state/$slug"
  mkdir -p "$root/t3_semantic" "$root/c4_semantic"
  mapfile -t formal < <(find "$state/download" -type f -path "*/$job/eval/formal_t3_300/predictions.jsonl")
  mapfile -t diag_pred < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
  mapfile -t diag_eval < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
  mapfile -t diag_metrics < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/metrics.json")
  [[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
    || fail "rescue artifact cardinality drift job=$job"
  attempt=0
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$root/t3_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$root/t3_semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
      >> "$root/t3_semantic/judge.log" 2>&1; do
    attempt=$((attempt+1)); (( attempt < 30 )) || fail "rescue T3 judge exhausted job=$job"
    heartbeat; sleep_unlocked 120
  done
  attempt=0
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
      --output-dir "$root/c4_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$root/c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
      >> "$root/c4_semantic/judge.log" 2>&1; do
    attempt=$((attempt+1)); (( attempt < 30 )) || fail "rescue C4 judge exhausted job=$job"
    heartbeat; sleep_unlocked 120
  done
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$root/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" \
    --output "$root/c4_semantic/summary.json" \
    >> "$root/c4_semantic/summarize.log" 2>&1 \
    || fail "rescue diagnosis C4 summary failed job=$job"
done

set +e
"$python_bin" scripts/select_state8_strong1_rescue.py \
  --root "$state" --output "$state/selection.json" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "rescue scientific selection crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "conditional strong1 rescue reached terminal scientific selection"
