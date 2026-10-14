#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/task_retention_rehearsal"
experiment="$(<"$state_dir/active_experiment.txt")"
config="$project_dir/amlt/feeds_layerwise_task_retention_rehearsal_20260801.yaml"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
mkdir -p "$state_dir/preserved" "$state_dir/t3_semantic"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

recovery_wave=0
if [[ "$experiment" =~ -recovery([0-9]+)-seed42- ]]; then
  recovery_wave="${BASH_REMATCH[1]}"
fi
expected_jobs=2
if [[ -f "$state_dir/active_jobs_experiment.txt" \
      && "$(<"$state_dir/active_jobs_experiment.txt")" == "$experiment" \
      && -s "$state_dir/active_jobs.txt" ]]; then
  expected_jobs="$(wc -l < "$state_dir/active_jobs.txt")"
fi

wait_quota(){
  local required="$1" output="$2" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" "$output" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); n=int(sys.argv[2]); q=row['quotas_(used/limit)']
available=int(q['standard']['limit'])-int(q['standard']['used'])
user_available=int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)
open(sys.argv[3],'w',encoding='utf-8').write(json.dumps({'selected_row':row,'required_units':n,'standard_available':available,'user_available':user_available},indent=2,sort_keys=True)+'\n')
assert available>=n and user_available>=n,(available,user_available,n)
PY
    then return 0; fi
    sleep 120
  done
}
while true; do
  heartbeat
  status="$(amlt status "$experiment" 2>&1)"
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|InternalServerError|ServiceUnavailable' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then break; fi
  amlt results download "$experiment" --status pass \
    --include '*.json' --include 'predictions.jsonl' --include 'predictions_redacted.jsonl' \
    --output "$state_dir/preserved" >> "$state_dir/controller.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal task-retention state has no failed sibling"
  (( recovery_wave < 6 )) || fail "task-retention rehearsal exhausted six recoveries"
  recovery_wave=$((recovery_wave+1))
  selectors=(); for job in "${failed_jobs[@]}"; do selectors+=(":$job"); done
  next_experiment="layerwise-task-retention-rehearsal-nccl2h-recovery${recovery_wave}-seed42-20260802"
  wait_quota "$((4 * ${#failed_jobs[@]}))" "$state_dir/${next_experiment}.quota.json"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${next_experiment}.preflight.dump.yaml" \
    2> "$state_dir/${next_experiment}.preflight.stderr" \
    || { flock -u 8; fail "task-retention recovery dump failed"; }
  python scripts/validate_layerwise_task_retention_rehearsal_amlt.py \
    --config "$state_dir/${next_experiment}.preflight.dump.yaml" \
    --jobs "${failed_jobs[@]}" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "task-retention recovery preflight failed"; }
  current="$(amlt status --no-update-db "$next_experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "could not classify task-retention recovery"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "${selectors[@]}" "$next_experiment" -t Feeds -y \
      --description "Failed-sibling recovery wave=$recovery_wave with validated 7200-second ProcessGroupNCCL timeout; scientific recipe unchanged" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${next_experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; fail "task-retention recovery submission failed"; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${next_experiment}.submission.txt"
  else
    flock -u 8; fail "ambiguous task-retention recovery state"
  fi
  flock -u 8
  experiment="$next_experiment"
  expected_jobs="${#failed_jobs[@]}"
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
  printf '%s\n' "$experiment" > "$state_dir/active_jobs_experiment.txt"
  printf '%s\n' "${failed_jobs[@]}" > "$state_dir/active_jobs.txt"
  log "submitted isolated failed-sibling recovery wave=$recovery_wave jobs=${failed_jobs[*]} experiment=$experiment"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved/." "$state_dir/download/" 2>/dev/null || true
amlt results download "$experiment" --status pass \
  --include '*.json' --include 'predictions.jsonl' --include 'predictions_redacted.jsonl' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "task-retention result download failed"
readarray -t predictions < <(find "$state_dir/download" -type f \
  -path '*/eval/formal_screen/predictions.jsonl' | sort)
[[ ${#predictions[@]} -eq 2 ]] || fail "expected two task-retention formal prediction files"

python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
backend=responses
extra="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  python_bin=/home/v-yuejiali/fkft/.venv/bin/python
  backend=trapi
  extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi
for prediction in "${predictions[@]}"; do
  if [[ "$prediction" == *equal-t3-t6* ]]; then branch=equal_t3_t6; else branch=t3_heavy; fi
  output="$state_dir/t3_semantic/$branch"
  mkdir -p "$output"
  heartbeat
  PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" \
    --output-dir "$output" --judge-backend "$backend" --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
    --judge-audit-samples 75 --seed 42 >> "$state_dir/t3_semantic/$branch.log" 2>&1 \
    || fail "task-retention T3 judge failed branch=$branch"
  "$python_bin" scripts/validate_checkpoint_t3_semantic_result.py \
    --metrics "$output/metrics.json" --output "$output/gate.json" --checkpoint-round 750 \
    >> "$state_dir/controller.log" 2>&1 || true
done

python scripts/summarize_layerwise_task_retention_rehearsal.py \
  --root "$state_dir/download" --t3-root "$state_dir/t3_semantic" \
  --output "$state_dir/comparison.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "task-retention comparison failed"
touch "$state_dir/complete"
log "task-retention rehearsal comparison complete"
