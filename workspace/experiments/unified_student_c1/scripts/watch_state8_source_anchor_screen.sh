#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_source_anchor_screen_20260811"
config="$project/amlt/feeds_state8_source_anchor_screen_20260811.yaml"
base=state8-source-anchor-screen-20260811
recovery_prefix=state8-source-anchor-screen-recovery
experiment="$base"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
jobs=(
  state8-source-anchor-full-mse1-seed3071
  state8-source-anchor-updater-mse1-seed3072
  state8-source-anchor-full-mse2-seed3073
)
arms=(full_mse1 updater_mse1 full_mse2)
max_waves="${MAX_RECOVERY_WAVES:-12}"
wave=0
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"

mkdir -p "$state/preserved_results"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260811$ ]] && wave="${BASH_REMATCH[1]}"
else
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
fi

preserve_passed(){
  local current="$1" status="$2" job
  mapfile -t passed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
    if str(row.get('status') or '').casefold() == 'pass':
        print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${passed[@]}"; do
    [[ -f "$state/passed.$job" ]] && continue
    amlt results download "$current" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 \
      || return 1
    touch "$state/passed.$job"
    log "preserved job=$job experiment=$current"
  done
}

submit_missing(){
  local next="$1"; shift
  local missing=("$@") selectors=() output rc current class
  for job in "${missing[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$next.preflight.yaml" 2> "$state/$next.preflight.stderr" \
    || { flock -u 8; return 1; }
  validator_args=()
  for job in "${missing[@]}"; do validator_args+=(--job "$job"); done
  "$python_bin" scripts/validate_state8_source_anchor_screen_amlt.py \
    --config "$state/$next.preflight.yaml" "${validator_args[@]}" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$next" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" \
      2>> "$state/controller.log" || { flock -u 8; return 1; }
    output="$(amlt run "$config" "${selectors[@]}" "$next" -t Feeds -y \
      --description "Automatic hash-bound recovery for State8 source-anchor unified-model screen" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$next.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$next" > "$state/active_experiment.txt"
}

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" "$status" || fail "result preservation failed"
  missing=()
  for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  (( ${#missing[@]} == 0 )) && { touch "$state/remote_passed"; break; }
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
states={str(row.get('status') or '').casefold() for row in json.loads(os.environ['STATUS'])}
active={'preparing','pre-queued','queued','running','resuming'}
if states & active: print('active')
elif 'paused' in states: print('paused')
elif states: print('terminal')
else: print('invalid')
PY
)"
  case "$phase" in
    active) sleep_unlocked 60 ;;
    paused)
      mapfile -t paused < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
    if str(row.get('status') or '').casefold() == 'paused':
        print(':'+str(row['job_name']).lstrip(':'))
PY
      )
      timeout 90 amlt resume "$experiment" "${paused[@]}" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_waves )) || fail "source-anchor recovery waves exhausted"
      wave=$((wave+1)); next="${recovery_prefix}${wave}-20260811"
      until submit_missing "$next" "${missing[@]}"; do heartbeat; sleep_unlocked 120; done
      experiment="$next"
      log "submitted recovery wave=$wave jobs=${missing[*]}" ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
cp -a "$state/preserved_results/." "$state/download/" \
  || fail "source-anchor preserved-result staging failed"

candidates=()
for index in "${!jobs[@]}"; do
  job="${jobs[$index]}"; arm="${arms[$index]}"; out="$state/semantic/$arm"
  mapfile -t roots < <(find "$state/download" -type d -name "$job")
  [[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality arm=$arm"
  root="${roots[0]}"
  mapfile -t formal_pred < <(find "$root" -type f -path '*/eval/formal_t3_300/predictions.jsonl')
  mapfile -t formal_metrics < <(find "$root" -type f -path '*/eval/formal_t3_300/metrics.json')
  mapfile -t diag_pred < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/judge_predictions.jsonl')
  mapfile -t diag_eval < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/judge_eval.jsonl')
  mapfile -t diag_metrics < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/metrics.json')
  mapfile -t source_receipt < <(find "$root" -type f -path '*/metrics/diagnosis_first_source_anchored_repair.json')
  mapfile -t completion < <(find "$root" -type f -path '*/metrics/completion.json')
  [[ ${#formal_pred[@]} -eq 1 && ${#formal_metrics[@]} -eq 1 \
     && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 \
     && ${#source_receipt[@]} -eq 1 && ${#completion[@]} -eq 1 ]] \
    || fail "evaluation artifact cardinality arm=$arm"
  mkdir -p "$out/t3" "$out/c4"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${formal_pred[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$out/t3" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$out/t3/cache.jsonl" --judge-audit-samples 75 --seed 42 \
      >> "$out/t3/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
      --output-dir "$out/c4" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$out/c4/cache.jsonl" --judge-audit-samples 100 --seed 42 \
      >> "$out/c4/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$out/c4/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" --output "$out/c4/summary.json" \
    >> "$out/c4/summarize.log" 2>&1 || fail "diagnosis-C4 summary failed arm=$arm"
  "$python_bin" scripts/summarize_state8_source_anchor_candidate.py \
    --arm "$arm" --job "$job" --formal-metrics "${formal_metrics[0]}" \
    --t3-semantic "$out/t3/metrics.json" --diagnosis-c4 "$out/c4/summary.json" \
    --source-receipt "${source_receipt[0]}" --completion "${completion[0]}" \
    --output "$out/summary.json" || fail "candidate summary failed arm=$arm"
  candidates+=(--candidate "$out/summary.json")
done

set +e
"$python_bin" scripts/select_state8_source_anchor_candidate.py \
  "${candidates[@]}" --output "$state/selection.json" >> "$state/controller.log" 2>&1
selection_rc=$?
set -e
if (( selection_rc == 0 )); then touch "$state/diagnostic_gate_pass"
elif (( selection_rc == 2 )); then touch "$state/diagnostic_gate_failed"
else fail "source-anchor selector crashed rc=$selection_rc"; fi
if (( selection_rc == 2 )); then
  touch "$state/complete"; heartbeat
  log "source-anchor semantic screen complete without a diagnostic-qualified arm"
  exit 0
fi

answer_template="$project/amlt/feeds_state8_source_anchor_answer_c4_20260811.yaml"
answer_config="$state/answer_c4.generated.yaml"
answer_base=state8-source-anchor-answer-c4-20260811
answer_recovery_prefix=state8-source-anchor-answer-c4-recovery
answer_experiment="$answer_base"
answer_job=state8-source-anchor-answer-c4-selected
answer_wave=0
"$python_bin" scripts/materialize_state8_source_anchor_answer_c4.py \
  --template "$answer_template" --selection "$state/selection.json" --output "$answer_config" \
  || fail "answer-C4 config materialization failed"
"$python_bin" scripts/validate_state8_source_anchor_answer_c4_amlt.py --config "$answer_config" \
  >> "$state/controller.log" 2>&1 || fail "answer-C4 config validation failed"

submit_answer_c4(){
  local next="$1" output rc current class
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$answer_config" --dump -t Feeds > "$state/$next.preflight.yaml" \
    2> "$state/$next.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_source_anchor_answer_c4_amlt.py \
    --config "$state/$next.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$next" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" \
      2>> "$state/controller.log" || { flock -u 8; return 1; }
    output="$(amlt run "$answer_config" "$next" -t Feeds -y \
      --description "Frozen 150-pair answer-C4 confirmation for diagnostic-selected State8 unified candidate" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$next.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8
  printf '%s\n' "$next" > "$state/answer_c4_active_experiment.txt"
}

if [[ -s "$state/answer_c4_active_experiment.txt" ]]; then
  answer_experiment="$(<"$state/answer_c4_active_experiment.txt")"
  [[ "$answer_experiment" =~ recovery([0-9]+)-20260811$ ]] && answer_wave="${BASH_REMATCH[1]}"
else
  until submit_answer_c4 "$answer_experiment"; do heartbeat; sleep_unlocked 120; done
  log "submitted frozen answer-C4 experiment=$answer_experiment"
fi

while [[ ! -f "$state/answer_c4_remote_passed" ]]; do
  heartbeat
  answer_status="$(timeout 90 amlt --json-tables status "$answer_experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$answer_status" && "$answer_status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$answer_status" > "$state/answer_c4_latest_status.json"
  answer_phase="$(STATUS="$answer_status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); states={str(x.get('status') or '').casefold() for x in rows}
if len(rows)==1 and states=={'pass'}: print('pass')
elif states & {'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in states: print('paused')
else: print('terminal')
PY
)"
  case "$answer_phase" in
    pass) touch "$state/answer_c4_remote_passed" ;;
    active) sleep_unlocked 60 ;;
    paused)
      timeout 90 amlt resume "$answer_experiment" ":$answer_job" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    *)
      timeout 90 amlt show "$answer_experiment" > "$state/$answer_experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$answer_experiment" ":$answer_job" \
        > "$state/$answer_experiment.failure.log" 2>&1 || true
      (( answer_wave < max_waves )) || fail "answer-C4 recovery waves exhausted"
      answer_wave=$((answer_wave+1)); answer_experiment="${answer_recovery_prefix}${answer_wave}-20260811"
      until submit_answer_c4 "$answer_experiment"; do heartbeat; sleep_unlocked 120; done
      log "submitted answer-C4 recovery wave=$answer_wave" ;;
  esac
done

rm -rf "$state/answer_c4_download"; mkdir -p "$state/answer_c4_download"
amlt results download "$answer_experiment" ":$answer_job" \
  --include '*.json' --include '*.txt' --output "$state/answer_c4_download" \
  >> "$state/controller.log" 2>&1 || fail "answer-C4 result download failed"
mapfile -t answer_json < <(find "$state/answer_c4_download" -type f -path '*/eval/answer_c4_seed42_n150/recurrent_causal_gate.json')
[[ ${#answer_json[@]} -eq 1 ]] || fail "answer-C4 artifact cardinality drift"
set +e
"$python_bin" scripts/finalize_state8_source_anchor_answer_c4.py \
  --selection "$state/selection.json" --answer-c4 "${answer_json[0]}" \
  --output "$state/joint_pilot_gate.json" >> "$state/controller.log" 2>&1
joint_rc=$?
set -e
if (( joint_rc == 0 )); then touch "$state/joint_pilot_gate_pass"
elif (( joint_rc == 2 )); then touch "$state/joint_pilot_gate_failed"
else fail "joint pilot gate finalizer crashed rc=$joint_rc"; fi
touch "$state/complete"; heartbeat
log "source-anchor semantic screen and frozen answer-C4 joint pilot gate complete"
