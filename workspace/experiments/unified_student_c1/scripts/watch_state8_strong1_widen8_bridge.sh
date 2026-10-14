#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
train_experiment=state8-strong1-widen8-bridge-20260808
train_recovery_prefix=state8-strong1-widen8-bridge-recovery
train_recovery_suffix=-20260808
max_train_recovery_waves=16
eval_experiment=state8-strong1-widen8-eval-20260808
eval_recovery_prefix=state8-strong1-widen8-eval-recovery
eval_recovery_suffix=-20260808
max_eval_recovery_waves=16
eval_config="$project/amlt/feeds_state8_strong1_widen8_eval_20260808.yaml"
state="$project/results/state8_strong1_widen8_bridge_20260808"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
# The seed58/lr5e-7 arm was retired before training to keep this account below
# its 32-GPU active-request limit.  Its learning-rate family remains covered by
# the independent one-A100 seed60 fallback; the primary bridge retains seed59.
jobs=(state8-strong1-widen8-bridge-lr1e6-seed59)
eval_jobs=(state8-strong1-widen8-eval-lr1e6-seed59)
slugs=(lr1e6_seed59)

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
heartbeat
(while true; do heartbeat; sleep 30; done) &
heartbeat_pid=$!

superseded(){
  local selection="$state/complete_pipeline/selection.json"
  [[ -s "$selection" ]] || return 1
  "$python_bin" - "$selection" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); slug=str(x['selected']['slug'])
raise SystemExit(1 if slug=='lr1e6_seed59' else 0)
PY
}

status_json(){ timeout 90 amlt --json-tables status "$1" --hide-urls 2>/dev/null; }
status_class(){
  STATUS_JSON="$1" EXPECTED="$2" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS_JSON']); expected=int(os.environ['EXPECTED'])
if len(rows)!=expected: print('invalid')
else:
 s=[str(x.get('status') or '').casefold() for x in rows]
 if all(x=='pass' for x in s): print('pass')
 elif any(x in {'preparing','pre-queued','queued','running','paused','resuming'} for x in s): print('active')
 elif any(x in {'failed','fail','cancelled','canceled','stopped','killed'} for x in s): print('failed')
 else: print('unknown')
PY
}

observe_training(){
  local experiment="$1" job="$2" root="$state/progress/$job" listing log_file
  mkdir -p "$root"
  listing="$(timeout 90 amlt logs list "$experiment" ":$job" 2>/dev/null)" || return 0
  log_file="$(LISTING="$listing" "$python_bin" - <<'PY'
import os,re
paths=[]
for line in os.environ['LISTING'].splitlines():
 value=line.strip()
 if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
  retry=re.search(r'retry_(\d+)',value); paths.append((int(retry.group(1)) if retry else -1,value))
print(max(paths)[1] if paths else '')
PY
)" || return 0
  [[ -n "$log_file" ]] || return 0
  timeout 90 amlt logs view -F "$log_file" -n 800 "$experiment" ":$job" \
    > "$root/latest_rank0.log" 2> "$root/latest_rank0.stderr" || return 0
  "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
    --log "$root/latest_rank0.log" --experiment "$experiment:$job" \
    --log-file "$log_file" --status running --output "$root/progress.json" \
    --total-rounds 2500 --max-run-seconds 43200 >/dev/null 2>&1 || true
}

preserve_passed_training(){
  local experiment="$1" status pass_count
  status="$(status_json "$experiment")" || return 1
  pass_count="$(STATUS_JSON="$status" "$python_bin" - <<'PY'
import json,os
print(sum(str(row.get('status') or '').casefold()=='pass'
          for row in json.loads(os.environ['STATUS_JSON'])))
PY
)" || return 1
  (( pass_count > 0 )) || return 0
  mkdir -p "$state/train_preserved_results"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/train_preserved_results" >> "$state/controller.log" 2>&1
}

preserve_passed_eval(){
  local experiment="$1" status pass_count
  status="$(status_json "$experiment")" || return 1
  pass_count="$(STATUS_JSON="$status" "$python_bin" - <<'PY'
import json,os
print(sum(str(row.get('status') or '').casefold()=='pass'
          for row in json.loads(os.environ['STATUS_JSON'])))
PY
)" || return 1
  (( pass_count > 0 )) || return 0
  mkdir -p "$state/eval_preserved_results"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/eval_preserved_results" >> "$state/controller.log" 2>&1
}

submit_eval_recovery(){
  local experiment="$1"; shift
  local failed=("$@") selectors=() job
  for job in "${failed[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$eval_config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_strong1_widen8_eval_amlt.py \
    --config "$state/$experiment.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  amlt run "$eval_config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Atomic recovery of failed strong1-to-State8 strict evaluation arms" \
    > "$state/$experiment.submission.txt" 2>&1
  rc=$?; flock -u 8; return "$rc"
}

submit_training_recovery(){
  local experiment="$1"; shift
  local failed=("$@") selectors=() job
  for job in "${failed[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run amlt/feeds_state8_strong1_widen8_bridge_20260808.yaml \
    "${selectors[@]}" --dump -t Feeds > "$state/$experiment.preflight.yaml" \
    2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_strong1_widen8_bridge_amlt.py \
    --config "$state/$experiment.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  amlt run amlt/feeds_state8_strong1_widen8_bridge_20260808.yaml \
    "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Atomic recovery of failed strong1-to-State8 bridge arms; exact remote checkpoint resume" \
    > "$state/$experiment.submission.txt" 2>&1
  rc=$?; flock -u 8; return "$rc"
}

if [[ ! -s "$state/train_active_experiment.txt" ]]; then
  printf '%s\n' "$train_experiment" > "$state/train_active_experiment.txt"
  printf '%s\n' "${jobs[@]}" > "$state/train_active_jobs.txt"
fi
train_experiment="$(<"$state/train_active_experiment.txt")"
train_wave=0
[[ "$train_experiment" =~ recovery([0-9]+)-20260808 ]] && train_wave="${BASH_REMATCH[1]}"
while [[ ! -f "$state/train_remote_passed" ]]; do
  heartbeat
  status="$(status_json "$train_experiment")" || { sleep 120; continue; }
  printf '%s\n' "$status" > "$state/train_latest_status.json"
  expected="$(wc -l < "$state/train_active_jobs.txt")"
  class="$(status_class "$status" "$expected")" || { sleep 120; continue; }
  while IFS= read -r job; do
    if STATUS_JSON="$status" JOB=":$job" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS_JSON']):
 if row.get('job_name')==os.environ['JOB'] and str(row.get('status')).casefold()=='running': raise SystemExit(0)
raise SystemExit(1)
PY
    then observe_training "$train_experiment" "$job"; fi
  done < "$state/train_active_jobs.txt"
  if [[ "$class" == active ]]; then sleep 120; continue; fi
  preserve_passed_training "$train_experiment" \
    || fail "could not preserve passed training arms experiment=$train_experiment"
  if [[ "$class" == pass ]]; then
    while IFS= read -r job; do touch "$state/train_passed.$job"; done < "$state/train_active_jobs.txt"
    all_passed=1
    for job in "${jobs[@]}"; do [[ -f "$state/train_passed.$job" ]] || all_passed=0; done
    if (( all_passed )); then touch "$state/train_remote_passed"; break; fi
    fail "training pass accounting lost a required arm"
  fi
  [[ "$class" == failed ]] || fail "training reached terminal/nonstandard class=$class"
  if superseded; then
    touch "$state/superseded" "$state/complete"
    log "short primary candidate became redundant; skipping recovery after terminal state"
    exit 0
  fi
  mapfile -t passed < <(STATUS_JSON="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS_JSON']):
 if str(row.get('status')).casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${passed[@]}"; do touch "$state/train_passed.$job"; done
  mapfile -t failed < <(STATUS_JSON="$status" "$python_bin" - <<'PY'
import json,os
terminal={'failed','fail','cancelled','canceled','stopped','killed'}
for row in json.loads(os.environ['STATUS_JSON']):
 if str(row.get('status')).casefold() in terminal: print(str(row['job_name']).lstrip(':'))
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal training state had no failed arms"
  (( train_wave < max_train_recovery_waves )) || fail "training exhausted recovery waves"
  train_wave=$((train_wave+1))
  train_experiment="${train_recovery_prefix}${train_wave}${train_recovery_suffix}"
  submit_training_recovery "$train_experiment" "${failed[@]}" \
    || fail "training recovery submission failed wave=$train_wave"
  printf '%s\n' "$train_experiment" > "$state/train_active_experiment.txt"
  printf '%s\n' "${failed[@]}" > "$state/train_active_jobs.txt"
  log "submitted training recovery wave=$train_wave experiment=$train_experiment jobs=${failed[*]}"
done

if superseded; then
  touch "$state/superseded" "$state/complete"
  log "short primary candidate became redundant after training; skipping evaluation"
  exit 0
fi

if [[ ! -f "$state/t3_semantic_complete" ]]; then
  rm -rf "$state/train_download"; mkdir -p "$state/train_download"
  cp -a "$state/train_preserved_results/." "$state/train_download/" \
    || fail "training preserved-result materialization failed"
  [[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
  for index in "${!jobs[@]}"; do
    job="${jobs[$index]}"; slug="${slugs[$index]}"; root="$state/$slug"
    mkdir -p "$root/t3_semantic"
    mapfile -t predictions < <(find "$state/train_download" -type f \
      -path "*/$job/eval/formal_screen/predictions.jsonl")
    [[ ${#predictions[@]} -eq 1 ]] || fail "formal prediction cardinality drift job=$job"
    attempt=0
    until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
        --output-dir "$root/t3_semantic" --judge-backend trapi \
        --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$root/t3_semantic/cache.jsonl" --judge-audit-samples 100 \
        --seed 42 >> "$root/t3_semantic/judge.log" 2>&1; do
      attempt=$((attempt+1)); (( attempt < 30 )) \
        || fail "T3 semantic judge exhausted job=$job"
      heartbeat; sleep 120
    done
  done
  touch "$state/t3_semantic_complete"
fi

if [[ ! -f "$state/eval_submitted" ]]; then
  "$python_bin" scripts/validate_state8_strong1_widen8_eval_amlt.py \
    --config "$eval_config" >> "$state/controller.log" 2>&1 \
    || fail "dependent evaluation source config invalid"
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  eval_selectors=(); for job in "${eval_jobs[@]}"; do eval_selectors+=(":$job"); done
  amlt run "$eval_config" "${eval_selectors[@]}" --dump -t Feeds > "$state/eval_preflight.yaml" \
    2> "$state/eval_preflight.stderr" || { flock -u 8; fail "evaluation AMLT dump failed"; }
  amlt --json-tables ti sing -t Feeds > "$state/eval_quota_at_submit.json" \
    2>> "$state/controller.log" || { flock -u 8; fail "evaluation quota check failed"; }
  amlt run "$eval_config" "${eval_selectors[@]}" "$eval_experiment" -t Feeds -y \
    --description "Strict state-only T1-T6 follow-up: answer C4 plus natural diagnosis C4 for both strong1-to-State8 bridge arms" \
    > "$state/eval_submission.txt" 2>&1 \
    || { flock -u 8; fail "dependent evaluation submission failed"; }
  flock -u 8
  touch "$state/eval_submitted"
  printf '%s\n' "$eval_experiment" > "$state/eval_active_experiment.txt"
  printf '%s\n' "${eval_jobs[@]}" > "$state/eval_active_jobs.txt"
  log "submitted dependent semantic-C4 evaluation experiment=$eval_experiment"
fi

eval_wave=0
if [[ -s "$state/eval_active_experiment.txt" ]]; then
  eval_experiment="$(<"$state/eval_active_experiment.txt")"
  [[ "$eval_experiment" =~ recovery([0-9]+)-20260808 ]] && eval_wave="${BASH_REMATCH[1]}"
fi
while [[ ! -f "$state/eval_remote_passed" ]]; do
  heartbeat
  status="$(status_json "$eval_experiment")" || { sleep 120; continue; }
  printf '%s\n' "$status" > "$state/eval_latest_status.json"
  expected="$(wc -l < "$state/eval_active_jobs.txt")"
  class="$(status_class "$status" "$expected")" || { sleep 120; continue; }
  if [[ "$class" == active ]]; then sleep 120; continue; fi
  preserve_passed_eval "$eval_experiment" \
    || fail "could not preserve passed evaluation arms experiment=$eval_experiment"
  mapfile -t passed < <(STATUS_JSON="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS_JSON']):
 if str(row.get('status')).casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${passed[@]}"; do touch "$state/eval_passed.$job"; done
  if [[ "$class" == pass ]]; then
    all_passed=1
    for job in "${eval_jobs[@]}"; do [[ -f "$state/eval_passed.$job" ]] || all_passed=0; done
    (( all_passed )) || fail "evaluation pass accounting lost a required arm"
    touch "$state/eval_remote_passed"; break
  fi
  [[ "$class" == failed ]] || fail "dependent evaluation reached terminal/nonstandard class=$class"
  mapfile -t failed < <(STATUS_JSON="$status" "$python_bin" - <<'PY'
import json,os
terminal={'failed','fail','cancelled','canceled','stopped','killed'}
for row in json.loads(os.environ['STATUS_JSON']):
 if str(row.get('status')).casefold() in terminal: print(str(row['job_name']).lstrip(':'))
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal evaluation state had no failed arms"
  (( eval_wave < max_eval_recovery_waves )) || fail "evaluation exhausted recovery waves"
  eval_wave=$((eval_wave+1))
  eval_experiment="${eval_recovery_prefix}${eval_wave}${eval_recovery_suffix}"
  submit_eval_recovery "$eval_experiment" "${failed[@]}" \
    || fail "evaluation recovery submission failed wave=$eval_wave"
  printf '%s\n' "$eval_experiment" > "$state/eval_active_experiment.txt"
  printf '%s\n' "${failed[@]}" > "$state/eval_active_jobs.txt"
  log "submitted evaluation recovery wave=$eval_wave jobs=${failed[*]}"
done

if [[ ! -f "$state/semantic_c4_complete" ]]; then
  rm -rf "$state/eval_download"; mkdir -p "$state/eval_download"
  cp -a "$state/eval_preserved_results/." "$state/eval_download/" \
    || fail "evaluation preserved-result materialization failed"
  for index in "${!slugs[@]}"; do
    slug="${slugs[$index]}"; job="${eval_jobs[$index]}"; root="$state/$slug"
    mkdir -p "$root/c4_semantic"
    mapfile -t judge_predictions < <(find "$state/eval_download" -type f \
      -path "*/$job/eval/diagnosis_c4_seed42_n50/judge_predictions.jsonl")
    mapfile -t judge_eval < <(find "$state/eval_download" -type f \
      -path "*/$job/eval/diagnosis_c4_seed42_n50/judge_eval.jsonl")
    mapfile -t generation_metrics < <(find "$state/eval_download" -type f \
      -path "*/$job/eval/diagnosis_c4_seed42_n50/metrics.json")
    [[ ${#judge_predictions[@]} -eq 1 && ${#judge_eval[@]} -eq 1 && ${#generation_metrics[@]} -eq 1 ]] \
      || fail "diagnosis-C4 artifact cardinality drift job=$job"
    attempt=0
    until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${judge_predictions[0]}" --eval-jsonl "${judge_eval[0]}" \
        --output-dir "$root/c4_semantic" --judge-backend trapi \
        --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$root/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
        --seed 42 >> "$root/c4_semantic/judge.log" 2>&1; do
      attempt=$((attempt+1)); (( attempt < 30 )) \
        || fail "C4 semantic judge exhausted job=$job"
      heartbeat; sleep 120
    done
    "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
      --judge-redacted "$root/c4_semantic/predictions_redacted.jsonl" \
      --generation-metrics "${generation_metrics[0]}" \
      --output "$root/c4_semantic/summary.json" \
      >> "$root/c4_semantic/summarize.log" 2>&1 \
      || fail "C4 semantic summary failed job=$job"
  done
  touch "$state/semantic_c4_complete"
fi

set +e
selection_args=()
for slug in "${slugs[@]}"; do selection_args+=(--candidate "$slug"); done
"$python_bin" scripts/select_state8_strong1_widen8_bridge.py \
  --root "$state" "${selection_args[@]}" \
  --output "$state/selection.json" > "$state/selection.log" 2>&1
selection_rc=$?
set -e
if (( selection_rc == 0 )); then
  touch "$state/scientific_gate_pass"
elif (( selection_rc == 2 )); then
  touch "$state/scientific_gate_failed"
else
  fail "joint bridge promotion selection crashed rc=$selection_rc"
fi
touch "$state/complete"; heartbeat
log "training, strict evaluation, and authoritative T3/C4 semantic judgments completed"
