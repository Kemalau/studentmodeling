#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
base_experiment=state8-diagnosis-causal-rescue-seed43-20260806
recovery_prefix=state8-diagnosis-causal-rescue-recovery
job=state8-diagnosis-causal-rescue-seed43
config="$project_dir/amlt/feeds_state8_diagnosis_causal_rescue_20260806.yaml"
state_dir="$project_dir/results/state8_diagnosis_causal_rescue_20260806"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"
max_recovery_waves="${MAX_RECOVERY_WAVES:-32}"

mkdir -p "$state_dir"
cd "$project_dir" || exit 1
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 30; continue; }
    if "$python_bin" - "$quota" "$state_dir/quota.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); q=row['quotas_(used/limit)']
standard_limit=int(q['standard']['limit']); standard_used=int(q['standard']['used'])
user_limit=int(q['user_max']['limit']); user_used=int(q['user_max'].get('used') or 0)
open(sys.argv[2],'w').write(json.dumps({'selected_row':row,'required':4,
 'standard_available':standard_limit-standard_used,
 'user_available':user_limit-user_used,'scheduler_queue_allowed':True},
 indent=2,sort_keys=True)+'\n')
assert standard_limit>=4 and user_limit>=4
PY
    then return 0; fi
    sleep 30
  done
}

submit_wave(){
  local experiment="$1" wave="$2" current rc class
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_diagnosis_causal_rescue_amlt.py \
    --config "$state_dir/${experiment}.preflight.yaml" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    wait_quota
    amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "Eight-state diagnosis/evidence causal rescue; exact checkpoint resume; full-history teacher to strict state-only student; semantic C4 twins; recovery_wave=$wave" \
      > "$state_dir/${experiment}.submission.txt" 2>&1 \
      || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
}

wave=0
if [[ -s "$state_dir/active_experiment.txt" ]]; then
  experiment="$(<"$state_dir/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-seed43 ]] && wave="${BASH_REMATCH[1]}"
else
  experiment="$base_experiment"
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep 120; continue; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then touch "$state_dir/remote_passed"; break; fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
    > "$state_dir/failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) || fail "diagnosis-causal rescue exhausted recovery waves"
  wave=$((wave+1))
  experiment="${recovery_prefix}${wave}-seed43-20260806"
  submit_wave "$experiment" "$wave" \
    || fail "diagnosis-causal rescue recovery submission failed wave=$wave"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "diagnosis-causal rescue result download failed"
readarray -t formal_predictions < <(find "$state_dir/download" -type f -path '*/eval/formal_screen/predictions.jsonl')
readarray -t diagnosis_predictions < <(find "$state_dir/download" -type f -path '*/eval/diagnosis_c4/judge_predictions.jsonl')
readarray -t diagnosis_evals < <(find "$state_dir/download" -type f -path '*/eval/diagnosis_c4/judge_eval.jsonl')
readarray -t diagnosis_metrics < <(find "$state_dir/download" -type f -path '*/eval/diagnosis_c4/metrics.json')
[[ ${#formal_predictions[@]} -eq 1 && ${#diagnosis_predictions[@]} -eq 1 \
   && ${#diagnosis_evals[@]} -eq 1 && ${#diagnosis_metrics[@]} -eq 1 ]] \
  || fail "downloaded result cardinality drift"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
mkdir -p "$state_dir/formal_semantic" "$state_dir/diagnosis_semantic"
attempt=0
while [[ ! -f "$state_dir/formal_semantic/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${formal_predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$state_dir/formal_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state_dir/formal_semantic/cache.jsonl" \
      --judge-audit-samples 100 --seed 43 >> "$state_dir/formal_semantic/judge.log" 2>&1; then
    touch "$state_dir/formal_semantic/complete"; break
  fi
  (( attempt < 30 )) || fail "formal semantic judge exhausted"
  sleep 120
done
attempt=0
while [[ ! -f "$state_dir/diagnosis_semantic/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${diagnosis_predictions[0]}" --eval-jsonl "${diagnosis_evals[0]}" \
      --output-dir "$state_dir/diagnosis_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state_dir/diagnosis_semantic/cache.jsonl" \
      --judge-audit-samples 100 --seed 42 >> "$state_dir/diagnosis_semantic/judge.log" 2>&1; then
    "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
      --judge-redacted "$state_dir/diagnosis_semantic/predictions_redacted.jsonl" \
      --generation-metrics "${diagnosis_metrics[0]}" \
      --output "$state_dir/diagnosis_semantic/summary.json" \
      >> "$state_dir/diagnosis_semantic/controller.log" 2>&1 \
      || fail "diagnosis semantic summary failed"
    touch "$state_dir/diagnosis_semantic/complete"; break
  fi
  (( attempt < 30 )) || fail "diagnosis semantic judge exhausted"
  sleep 120
done
touch "$state_dir/complete"; heartbeat
log "state8 diagnosis-causal rescue and both semantic judges completed"
