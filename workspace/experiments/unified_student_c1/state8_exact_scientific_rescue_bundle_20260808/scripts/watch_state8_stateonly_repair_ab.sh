#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
config="$project/amlt/feeds_state8_stateonly_repair_ab_20260806.yaml"
state="$project/results/state8_stateonly_repair_ab_20260806"
base=state8-stateonly-repair-ab-20260806
recovery_prefix=state8-stateonly-repair-ab-recovery
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
jobs=(state8-repair-logits-seed44 state8-repair-trajectory01-seed44)
max_waves="${MAX_RECOVERY_WAVES:-24}"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds

mkdir -p "$state/preserved_results"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

preserve_passed(){
  local experiment="$1"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/preserved_results" >> "$state/controller.log" 2>&1 || return 1
}

submit_jobs(){
  local experiment="$1"; shift; local names=("$@") current rc class
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${names[@]/#/:}" --dump -t Feeds \
    > "$state/${experiment}.preflight.yaml" 2> "$state/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_stateonly_repair_ab_amlt.py \
    --config "$state/${experiment}.preflight.yaml" --jobs "${names[@]}" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    amlt run "$config" "${names[@]/#/:}" "$experiment" -t Feeds -y \
      --description "State8 strict-state-only repair recovery; matched teacher-KD A/B; recovery=$experiment" \
      > "$state/${experiment}.submission.txt" 2>&1 || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state/active_jobs.txt"
}

experiment="$base"; wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260806 ]] && wave="${BASH_REMATCH[1]}"
else
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${jobs[@]}" > "$state/active_jobs.txt"
fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep 120; continue; fi
  expected="$(wc -l < "$state/active_jobs.txt")"
  if "$python_bin" - "$status" "$expected" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then preserve_passed "$experiment" || fail "could not preserve repair results"; break; fi
  preserve_passed "$experiment" || true
  mapfile -t paused < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
for name in paused_job_names(sys.argv[1]):
    print(name)
PY
  )
  if (( ${#paused[@]} )); then
    selectors=(); for job in "${paused[@]}"; do selectors+=(":$job"); done
    timeout 90 amlt resume "$experiment" "${selectors[@]}" >> "$state/controller.log" 2>&1 || true
    sleep 120; continue
  fi
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
for name in terminal_nonpass_job_names(sys.argv[1]):
    print(name)
PY
  )
  (( ${#failed[@]} )) || fail "terminal repair state without failed jobs"
  for job in "${failed[@]}"; do
    timeout 180 amlt logs view -n 2000 "$experiment" ":$job" \
      > "$state/${experiment}.${job}.failure.log" 2>&1 || true
  done
  (( wave < max_waves )) || fail "repair A/B exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260806"
  submit_jobs "$experiment" "${failed[@]}" || fail "repair recovery submission failed wave=$wave"
done

rm -rf "$state/download"; mkdir -p "$state/download"
cp -a "$state/preserved_results/." "$state/download/"
for arm in logits trajectory01; do
  job="state8-repair-${arm}-seed44"
  readarray -t formal < <(find "$state/download" -type f -path "*/$job/eval/formal_t3_300/predictions.jsonl")
  readarray -t diag_pred < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
  readarray -t diag_eval < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
  readarray -t diag_metrics < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/metrics.json")
  [[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
    || fail "repair artifact cardinality drift arm=$arm"
  mkdir -p "$state/semantic/$arm/t3" "$state/semantic/$arm/c4"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$state/semantic/$arm/t3" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/semantic/$arm/t3/cache.jsonl" \
    --judge-audit-samples 100 --seed 44 >> "$state/semantic/$arm/t3/judge.log" 2>&1 \
    || fail "T3 semantic judge failed arm=$arm"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
    --output-dir "$state/semantic/$arm/c4" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/semantic/$arm/c4/cache.jsonl" \
    --judge-audit-samples 100 --seed 42 >> "$state/semantic/$arm/c4/judge.log" 2>&1 \
    || fail "C4 semantic judge failed arm=$arm"
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$state/semantic/$arm/c4/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" \
    --output "$state/semantic/$arm/c4/summary.json" \
    >> "$state/semantic/$arm/c4/controller.log" 2>&1 \
    || fail "C4 semantic summary failed arm=$arm"
done

"$python_bin" scripts/select_state8_stateonly_repair_ab.py \
  --logits-t3 "$state/semantic/logits/t3/metrics.json" \
  --logits-c4 "$state/semantic/logits/c4/summary.json" \
  --trajectory-t3 "$state/semantic/trajectory01/t3/metrics.json" \
  --trajectory-c4 "$state/semantic/trajectory01/c4/summary.json" \
  --output "$state/selection.json" >> "$state/controller.log" 2>&1
rc=$?
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "repair selector failed operationally rc=$rc"; fi
touch "$state/complete"; heartbeat
log "state8 repair A/B completed"
