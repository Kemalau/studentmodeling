#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_strong1_widen8_bridge_20260808/single_fallback_monitor"
config="$project/amlt/feeds_state8_strong1_widen8_single_fallback_20260808.yaml"
job=state8-strong1-widen8-single-lr5e7-seed60
base=state8-strong1-widen8-single-fallback-recovery1-20260808
prefix=state8-strong1-widen8-single-fallback-recovery
suffix=-20260808
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
eval_config="$project/amlt/feeds_state8_strong1_widen8_single_eval_20260808.yaml"
eval_job=state8-strong1-widen8-single-eval-lr5e7-seed60
candidate_slug=single_lr5e7_seed60
candidate_root="$state/$candidate_slug"
max_recovery_waves=16

# Bind progress-risk accounting to the actual AMLT attempt slice.  This job
# deliberately uses resumable four-hour attempts; the broader recovery
# horizon is not the runtime available to the current attempt.
attempt_max_run_seconds="$($python_bin - "$config" "$job" <<'PY'
import sys,yaml
x=yaml.safe_load(open(sys.argv[1])); name=sys.argv[2]
matches=[row for row in x['jobs'] if row['name']==name]
assert len(matches)==1,matches
value=int(matches[0]['submit_args']['max_run_duration_seconds'])
assert 0 < value <= 86400,value
print(value)
PY
)" || exit 1

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
  # An operator may retire this queued legacy branch after a newer,
  # scientifically targeted branch has been submitted.  The immutable remote
  # checkpoints remain available for later recovery; this marker only prevents
  # the controller from immediately resubmitting work that was deliberately
  # removed from the congested queue.
  [[ -f "$state/retired_for_diagnosis_first_priority" ]] && return 0
  local selection="$state/../complete_pipeline/selection.json"
  [[ -s "$selection" ]] || return 1
  "$python_bin" - "$selection" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); slug=str(x['selected']['slug'])
raise SystemExit(1 if slug=='single_lr5e7_seed60' else 0)
PY
}

status_json(){ timeout 90 amlt --json-tables status "$1" --hide-urls 2>/dev/null; }
phase(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
print(str(rows[0].get('status') or '').casefold() if len(rows)==1 else 'invalid')
PY
}
observe(){
  local experiment="$1" root="$state/progress" listing log_file
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
    --total-rounds 5000 --max-run-seconds "$attempt_max_run_seconds" \
    >/dev/null 2>&1 || true
}
submit_recovery(){
  local experiment="$1"
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_strong1_widen8_single_fallback_amlt.py \
    --config "$config" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" --dump -t Feeds > "$state/$experiment.preflight.yaml" \
    2> "$state/$experiment.preflight.stderr" || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  amlt run "$config" "$experiment" -t Feeds -y \
    --description "Atomic recovery of full one-A100 GA4 strong1-to-State8 bridge; resumes newest valid GA=4-aligned checkpoint" \
    > "$state/$experiment.submission.txt" 2>&1
  rc=$?; flock -u 8; return "$rc"
}
submit_eval(){
  local eval_experiment="$1"
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_strong1_widen8_single_eval_amlt.py \
    --config "$eval_config" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$eval_config" --dump -t Feeds > "$state/$eval_experiment.preflight.yaml" \
    2> "$state/$eval_experiment.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_strong1_widen8_single_eval_amlt.py \
    --config "$state/$eval_experiment.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$eval_experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  amlt run "$eval_config" "$eval_experiment" -t Feeds -y \
    --description "Strict state-only answer-C4 and natural diagnosis-C4 for the full one-A100 GA4 strong1-to-State8 fallback" \
    > "$state/$eval_experiment.submission.txt" 2>&1
  rc=$?; flock -u 8; return "$rc"
}

experiment="$base"; wave=1
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260808 ]] && wave="${BASH_REMATCH[1]}"
fi
while true; do
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  heartbeat
  status="$(status_json "$experiment")" || { sleep 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  current="$(phase "$status")" || { sleep 120; continue; }
  case "$current" in
    pass) touch "$state/remote_passed"; break ;;
    preparing|pre-queued|queued|paused|resuming) sleep 120; continue ;;
    running) observe "$experiment"; sleep 120; continue ;;
    *)
      if superseded; then
        touch "$state/superseded" "$state/complete"
        log "single fallback became redundant; skipping recovery after terminal state"
        exit 0
      fi
      timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_recovery_waves )) || fail "single fallback exhausted recovery waves"
      wave=$((wave+1)); experiment="${prefix}${wave}${suffix}"
      submit_recovery "$experiment" || fail "single fallback recovery submission failed wave=$wave"
      log "submitted single fallback recovery wave=$wave experiment=$experiment"
      ;;
  esac
done
training_experiment="$experiment"

if superseded; then
  touch "$state/superseded" "$state/complete"
  log "single fallback became redundant after training; skipping evaluation"
  exit 0
fi

if [[ ! -f "$state/t3_semantic_complete" ]]; then
  rm -rf "$state/train_download"; mkdir -p "$state/train_download" "$candidate_root/t3_semantic"
  amlt results download "$training_experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/train_download" >> "$state/controller.log" 2>&1 \
    || fail "single fallback training-result download failed"
  mapfile -t predictions < <(find "$state/train_download" -type f \
    -path "*/$job/eval/formal_screen/predictions.jsonl")
  [[ ${#predictions[@]} -eq 1 ]] || fail "single fallback formal prediction cardinality drift"
  [[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
  attempt=0
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$candidate_root/t3_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$candidate_root/t3_semantic/cache.jsonl" --judge-audit-samples 100 \
      --seed 42 >> "$candidate_root/t3_semantic/judge.log" 2>&1; do
    attempt=$((attempt+1)); (( attempt < 30 )) \
      || fail "single fallback T3 semantic judge exhausted"
    heartbeat; sleep 120
  done
  touch "$state/t3_semantic_complete"
fi

eval_experiment=state8-strong1-widen8-single-eval-20260808
eval_wave=0
if [[ -s "$state/eval_active_experiment.txt" ]]; then
  eval_experiment="$(<"$state/eval_active_experiment.txt")"
  [[ "$eval_experiment" =~ recovery([0-9]+)-20260808 ]] && eval_wave="${BASH_REMATCH[1]}"
fi
while [[ ! -f "$state/eval_remote_passed" ]]; do
  if [[ ! -f "$state/eval_submitted.$eval_experiment" ]]; then
    submit_eval "$eval_experiment" || { sleep 120; continue; }
    printf '%s\n' "$eval_experiment" > "$state/eval_active_experiment.txt"
    touch "$state/eval_submitted.$eval_experiment"
    log "submitted single fallback evaluation experiment=$eval_experiment"
  fi
  heartbeat
  status="$(status_json "$eval_experiment")" || { sleep 120; continue; }
  printf '%s\n' "$status" > "$state/eval_latest_status.json"
  current="$(phase "$status")" || { sleep 120; continue; }
  case "$current" in
    pass) touch "$state/eval_remote_passed" ;;
    preparing|pre-queued|queued|paused|resuming|running) sleep 120 ;;
    *)
      timeout 90 amlt show "$eval_experiment" > "$state/$eval_experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$eval_experiment" ":$eval_job" \
        > "$state/$eval_experiment.failure.log" 2>&1 || true
      (( eval_wave < max_recovery_waves )) || fail "single fallback evaluation exhausted recovery waves"
      eval_wave=$((eval_wave+1))
      eval_experiment="state8-strong1-widen8-single-eval-recovery${eval_wave}-20260808"
      printf '%s\n' "$eval_experiment" > "$state/eval_active_experiment.txt"
      ;;
  esac
done

if [[ ! -f "$state/semantic_c4_complete" ]]; then
  rm -rf "$state/eval_download"; mkdir -p "$state/eval_download" "$candidate_root/c4_semantic"
  amlt results download "$eval_experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/eval_download" >> "$state/controller.log" 2>&1 \
    || fail "single fallback evaluation-result download failed"
  mapfile -t judge_predictions < <(find "$state/eval_download" -type f \
    -path "*/$eval_job/eval/diagnosis_c4_seed42_n50/judge_predictions.jsonl")
  mapfile -t judge_eval < <(find "$state/eval_download" -type f \
    -path "*/$eval_job/eval/diagnosis_c4_seed42_n50/judge_eval.jsonl")
  mapfile -t generation_metrics < <(find "$state/eval_download" -type f \
    -path "*/$eval_job/eval/diagnosis_c4_seed42_n50/metrics.json")
  [[ ${#judge_predictions[@]} -eq 1 && ${#judge_eval[@]} -eq 1 && ${#generation_metrics[@]} -eq 1 ]] \
    || fail "single fallback diagnosis-C4 artifact cardinality drift"
  attempt=0
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${judge_predictions[0]}" --eval-jsonl "${judge_eval[0]}" \
      --output-dir "$candidate_root/c4_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$candidate_root/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
      --seed 42 >> "$candidate_root/c4_semantic/judge.log" 2>&1; do
    attempt=$((attempt+1)); (( attempt < 30 )) \
      || fail "single fallback diagnosis-C4 semantic judge exhausted"
    heartbeat; sleep 120
  done
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$candidate_root/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${generation_metrics[0]}" \
    --output "$candidate_root/c4_semantic/summary.json" \
    >> "$candidate_root/c4_semantic/summarize.log" 2>&1 \
    || fail "single fallback diagnosis-C4 summary failed"
  touch "$state/semantic_c4_complete"
fi

set +e
"$python_bin" scripts/select_state8_strong1_widen8_bridge.py \
  --root "$state" --candidate "$candidate_slug" \
  --output "$state/selection.json" > "$state/selection.log" 2>&1
selection_rc=$?
set -e
if (( selection_rc == 0 )); then
  touch "$state/scientific_gate_pass"
elif (( selection_rc == 2 )); then
  touch "$state/scientific_gate_failed"
else
  fail "single fallback joint promotion selection crashed rc=$selection_rc"
fi
touch "$state/complete"; heartbeat
log "single fallback training and authoritative strict-state-only evaluation completed"
