#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_task_anchor_screen_20260808"
config="$project/amlt/feeds_state8_task_anchor_screen_20260808.yaml"
bundle="$project/task_anchor_bundle_20260808"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
base_experiment=state8-task-anchor-screen-20260808
recovery_prefix=state8-task-anchor-screen-recovery
recovery_suffix=-20260808
jobs=(state8-task-anchor-base-seed56 state8-task-anchor-c4strong-seed57)
slugs=(base c4strong)
posttrain_stall_seconds="${STATE8_TASK_ANCHOR_POSTTRAIN_STALL_SECONDS:-21600}"

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
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

observe_progress(){
  local experiment="$1" job="$2" root="$state/progress/$job"
  local listing log_file phase threshold
  mkdir -p "$root"
  listing="$(timeout 90 amlt logs list "$experiment" ":$job" 2>&1)" || return 0
  log_file="$(LISTING="$listing" "$python_bin" - <<'PY'
import os,re
paths=[]
for line in os.environ['LISTING'].splitlines():
 value=line.strip()
 if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
  retry=re.search(r'retry_(\d+)',value)
  paths.append((int(retry.group(1)) if retry else -1,value))
print(max(paths)[1] if paths else '')
PY
)" || return 0
  if [[ -n "$log_file" ]] && timeout 90 amlt logs view -F "$log_file" -n 800 \
      "$experiment" ":$job" > "$root/latest_rank0.log" \
      2> "$root/latest_rank0.stderr"; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --log "$root/latest_rank0.log" --experiment "$experiment:$job" \
      --log-file "$log_file" --status running --output "$root/progress.json" \
      --total-rounds 250 --max-run-seconds 172800 \
      >/dev/null 2>&1 || true
  else
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment:$job" \
      --log-file "$log_file" --status running --output "$root/progress.json" \
      --total-rounds 250 >/dev/null 2>&1 || true
  fi
  [[ -s "$root/progress.json" ]] || return 0
  phase="$("$python_bin" - "$root/progress.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('phase') or '')
PY
)"
  if [[ "$phase" == awaiting_first_training_record ]]; then threshold=3600
  else threshold=2700; fi
  if "$python_bin" - "$root/progress.json" "$threshold" "$posttrain_stall_seconds" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); health=x.get('health')
unhealthy=isinstance(health,dict) and health.get('healthy') is False
training_complete=(float(x.get('progress_fraction') or 0)>=1.0
                   or int(x.get('optimizer_step') or -1)>=int(x.get('total_rounds') or -1)>0)
stall_threshold=float(sys.argv[3] if training_complete else sys.argv[2])
stalled=float(x.get('seconds_since_progress') or 0)>=stall_threshold
raise SystemExit(0 if unhealthy or stalled else 1)
PY
  then
    cp "$root/progress.json" "$root/stalled_or_unhealthy.json"
    timeout 90 amlt show "$experiment" ":$job" > "$root/show.txt" 2>&1 || true
    timeout 180 amlt logs view -n 2500 "$experiment" ":$job" \
      > "$root/failure.log" 2>&1 || true
    if timeout 90 amlt cancel "$experiment" ":$job" -y \
        > "$root/cancel.txt" 2>&1; then
      log "cancelled stalled/unhealthy task-anchor job=$job for exact recovery"
    fi
  else
    rm -f "$root/stalled_or_unhealthy.json"
  fi
}

preserve_passed(){
  local experiment="$1" status_json pass_count
  status_json="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || return 1
  pass_count="$(STATUS_JSON="$status_json" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS_JSON'])
print(sum(str(row.get('status')).casefold()=='pass' for row in rows))
PY
)" || return 1
  (( pass_count > 0 )) || return 0
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/preserved_results" >> "$state/controller.log" 2>&1
}

submit_jobs(){
  local experiment="$1"; shift
  local names=("$@") selectors=() status rc class output
  for name in "${names[@]}"; do selectors+=(":$name"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" \
    2> "$state/$experiment.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_task_anchor_screen_amlt.py \
    --config "$state/$experiment.preflight.yaml" --bundle "$bundle" \
    --jobs "${names[@]}" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  status="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(PYTHONPATH=scripts "$python_bin" - "$rc" "$status" <<'PY'
import sys
from recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Exact task-anchor screen recovery jobs=${names[*]}" 2>&1)"; rc=$?
    printf '%s\n' "$output" > "$state/$experiment.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$status" \
      > "$state/$experiment.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state/active_jobs.txt"
  log "submitted/attached task-anchor experiment=$experiment jobs=${names[*]}"
}

if [[ ! -s "$state/active_experiment.txt" ]]; then
  printf '%s\n' "$base_experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${jobs[@]}" > "$state/active_jobs.txt"
fi
experiment="$(<"$state/active_experiment.txt")"; wave=0
[[ "$experiment" =~ recovery([0-9]+)-20260808 ]] && wave="${BASH_REMATCH[1]}"

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" \
    >> "$state/status.log"
  while IFS= read -r job; do
    if JOB=":$job" STATUS="$status" "$python_bin" - <<'PY'
import os,re
text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',os.environ['STATUS'])
job=re.escape(os.environ['JOB'])
raise SystemExit(0 if re.search(rf'(?m)^.*{job}.*\brunning\b',text,re.I) else 1)
PY
    then observe_progress "$experiment" "$job"; fi
  done < "$state/active_jobs.txt"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep 120; continue; fi
  mapfile -t paused < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
sys.stdout.write(''.join(value+'\n' for value in paused_job_names(sys.argv[1])))
PY
  )
  if (( ${#paused[@]} > 0 )); then
    for job in "${paused[@]}"; do
      timeout 90 amlt resume "$experiment" ":$job" \
        >> "$state/controller.log" 2>&1 || true
    done
    log "requested resume for paused task-anchor jobs=${paused[*]}"
    sleep 120; continue
  fi
  expected="$(wc -l < "$state/active_jobs.txt")"
  if "$python_bin" - "$status" "$expected" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then preserve_passed "$experiment" || fail "could not preserve passed jobs"; break; fi
  preserve_passed "$experiment" || fail "could not preserve passed siblings"
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
sys.stdout.write(''.join(value+'\n' for value in terminal_nonpass_job_names(sys.argv[1])))
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal status had no failed jobs"
  for job in "${failed[@]}"; do
    timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
      > "$state/$experiment.$job.failure.log" 2>&1 || true
  done
  (( wave < 32 )) || fail "task-anchor screen exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}${recovery_suffix}"
  submit_jobs "$experiment" "${failed[@]}" || fail "recovery submit failed wave=$wave"
done

rm -rf "$state/download"; mkdir -p "$state/download"
cp -a "$state/preserved_results/." "$state/download/"
for index in "${!jobs[@]}"; do
  job="${jobs[$index]}"; slug="${slugs[$index]}"; arm="$state/$slug"
  mkdir -p "$arm/t3_semantic" "$arm/c4_semantic"
  mapfile -t formal < <(find "$state/download" -type f -path "*/$job/eval/formal_t3_300/predictions.jsonl")
  mapfile -t diag_pred < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
  mapfile -t diag_eval < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
  mapfile -t diag_metrics < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/metrics.json")
  [[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
    || fail "artifact cardinality drift arm=$slug"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$arm/t3_semantic" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$arm/t3_semantic/cache.jsonl" --judge-audit-samples 100 \
    --seed 42 >> "$arm/t3_semantic/judge.log" 2>&1 \
    || fail "T3 semantic judge failed arm=$slug"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
    --output-dir "$arm/c4_semantic" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$arm/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
    --seed 42 >> "$arm/c4_semantic/judge.log" 2>&1 \
    || fail "C4 semantic judge failed arm=$slug"
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$arm/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" \
    --output "$arm/c4_semantic/summary.json" \
    >> "$arm/c4_semantic/controller.log" 2>&1 \
    || fail "C4 semantic summary failed arm=$slug"
done

set +e
"$python_bin" scripts/select_state8_task_anchor_screen.py \
  --candidate base "$state/base" --candidate c4strong "$state/c4strong" \
  --output "$state/selection.json" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "task-anchor selector failed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "task-anchor screen reached terminal semantic selection"
