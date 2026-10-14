#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
teacher_state="$project/results/state8_diagnosis_first_teacher_stage_a_20260810"
state="$project/results/state8_diagnosis_first_matched_curriculum_20260810"
config=amlt/feeds_state8_diagnosis_first_matched_curriculum_20260810.yaml
experiment=state8-diagnosis-first-matched-curriculum-20260810
recovery_prefix=state8-diagnosis-first-matched-curriculum-recovery
job=state8-diagnosis-first-matched-curriculum-seed67
max_recovery_waves=24
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
full_teacher="$project/results/state8_diagnosis_first_teacher_full_eval_20260810"
corrected_c4="$project/results/state8_diagnosis_first_matched_final_diagnosis_c4_corrected_20260810"
progress="$state/progress"
stall_seconds="${STATE8_MATCHED_CURRICULUM_STALL_SECONDS:-2700}"
startup_stall_seconds="${STATE8_MATCHED_CURRICULUM_STARTUP_STALL_SECONDS:-7200}"
posttrain_stall_seconds="${STATE8_MATCHED_CURRICULUM_POSTTRAIN_STALL_SECONDS:-21600}"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure

mkdir -p "$state" "$progress"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"; flock -n 9 || exit 0
echo $$ > "$state/watcher.active.pid"
cleanup(){ rm -f "$state/watcher.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/watcher.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }
rm -f "$state/needs_attention"

wave=0

submit_recovery(){
  local next="$1" output rc class
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_diagnosis_first_matched_curriculum_amlt.py \
    --config "$config" >> "$state/watcher.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" --dump -t Feeds \
    > "$state/$next.preflight.yaml" 2> "$state/$next.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_diagnosis_first_matched_curriculum_amlt.py \
    --config "$state/$next.preflight.yaml" >> "$state/watcher.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" "$next" -t Feeds -y \
    --description "Automatic resumable recovery of diagnosis-first State8 matched curriculum and semantic C4 generation" 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state/$next.submission.txt"
  if (( rc != 0 )); then
    output="$(amlt status --no-update-db "$next" 2>&1)"; rc=$?
    class="$("$python_bin" - "$rc" "$output" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
    [[ "$class" == exists ]] || { flock -u 8; return 1; }
  fi
  flock -u 8
}

observe_running_progress(){
  local status_text="$1" listing log_file threshold_rc
  # Forward-progress states also include preparing/queued/resuming.  Only a
  # backend-confirmed Running job is eligible for a scientific liveness
  # decision; authentication and log-transport failures are never evidence
  # that the trainer has stalled.
  grep -Eqi '\brunning\b' <<< "$status_text" || {
    rm -f "$progress/stalled.json" "$progress/health_alert.json"
    return 0
  }
  listing="$(timeout 90 amlt logs list "$experiment" ":$job" 2>/dev/null)" \
    || return 0
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
  [[ -n "$log_file" ]] || return 0
  timeout 90 amlt logs view -F "$log_file" -n 700 \
    "$experiment" ":$job" > "$progress/latest_rank0.log" \
    2> "$progress/latest_rank0.stderr" || return 0
  "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
    --log "$progress/latest_rank0.log" --experiment "$experiment" \
    --log-file "$log_file" --status running --output "$progress/progress.json" \
    --total-rounds 5000 --max-run-seconds 86400 \
    --validation-reserve-seconds 21600 >> "$state/watcher.log" 2>&1 \
    || return 0
  "$python_bin" - "$progress/progress.json" "$stall_seconds" \
    "$startup_stall_seconds" "$posttrain_stall_seconds" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); health=x.get('health') or {}
unhealthy=health.get('healthy') is False
complete=(float(x.get('progress_fraction') or 0)>=1.0
          or int(x.get('round') or -1)>=int(x.get('total_rounds') or 5000))
awaiting=x.get('phase')=='awaiting_first_training_record'
threshold=float(sys.argv[4] if complete else sys.argv[3] if awaiting else sys.argv[2])
stalled=float(x.get('seconds_since_progress') or 0)>=threshold
raise SystemExit(0 if stalled or unhealthy else 1)
PY
  threshold_rc=$?
  if (( threshold_rc != 0 )); then
    rm -f "$progress/stalled.json" "$progress/health_alert.json"
    return 0
  fi
  cp "$progress/progress.json" "$progress/stalled.json"
  timeout 90 amlt show "$experiment" > "$state/$experiment.stalled.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
    > "$state/$experiment.stalled.log" 2>&1 || true
  if timeout 90 amlt cancel "$experiment" ":$job" -y \
      > "$state/$experiment.stalled.cancel.txt" 2>&1; then
    log "cancelled stalled/unhealthy matched curriculum for exact checkpoint recovery experiment=$experiment"
    return 2
  fi
  fail "stalled matched-curriculum cancellation failed experiment=$experiment"
}

# Stage A owns the semantic gate and the exactly-once Stage B submission.
while [[ ! -f "$teacher_state/curriculum_submitted" ]]; do
  heartbeat
  if [[ -f "$teacher_state/needs_attention" || -f "$teacher_state/scientific_gate_failed" ]]; then
    fail "teacher gate stopped the matched curriculum"
  fi
  sleep_unlocked 60
done
if [[ -s "$teacher_state/active_curriculum_experiment.txt" ]]; then
  experiment="$(<"$teacher_state/active_curriculum_experiment.txt")"
fi
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
fi
if [[ "$experiment" =~ recovery([0-9]+)-20260810$ ]]; then
  wave="${BASH_REMATCH[1]}"
fi
printf '%s\n' "$experiment" > "$state/active_experiment.txt"
log "monitoring matched curriculum experiment=$experiment"

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep_unlocked 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then touch "$state/remote_passed"; break; fi
  observe_running_progress "$status"
  progress_rc=$?
  if (( progress_rc == 2 )); then
    sleep_unlocked 60
    continue
  fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep_unlocked 60; continue; fi
  if grep -Eqi '\bpaused\b' <<< "$status"; then
    timeout 90 amlt resume "$experiment" ":$job" >> "$state/watcher.log" 2>&1 || true
    sleep_unlocked 60
    continue
  fi
  terminal="$($python_bin - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print('\n'.join(terminal_nonpass_job_names(sys.argv[1])))
PY
)" || { sleep_unlocked 60; continue; }
  [[ "$terminal" == "$job" ]] || { sleep_unlocked 60; continue; }
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_recovery_waves )) \
    || fail "matched curriculum exhausted automatic recovery waves experiment=$experiment"
  wave=$((wave+1))
  next="${recovery_prefix}${wave}-20260810"
  until submit_recovery "$next"; do
    heartbeat
    log "transient matched-curriculum recovery submission failure wave=$wave; retrying"
    sleep_unlocked 120
  done
  experiment="$next"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted matched-curriculum recovery wave=$wave experiment=$experiment"
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/watcher.log" 2>&1 \
  || fail "matched curriculum result download failed"
mapfile -t t3_predictions < <(find "$state/download" -type f -path '*/eval/formal_screen/predictions.jsonl')
mapfile -t answer_c4 < <(find "$state/download" -type f -path '*/eval/answer_c4_150/recurrent_causal_gate.json')
[[ ${#t3_predictions[@]} -eq 1 && ${#answer_c4[@]} -eq 1 ]] \
  || fail "semantic evaluation artifact cardinality drift"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "formal evaluation reference drift"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"

mkdir -p "$state/t3_semantic" "$state/c4_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${t3_predictions[0]}" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state/t3_semantic" --judge-backend trapi \
  --trapi-deployment gpt-4o_2024-11-20 \
  --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed 67 \
  >> "$state/t3_semantic/judge.log" 2>&1 || fail "T3 semantic judge failed"
# The training job was submitted before the diagnosis-C4 taxonomy-contract
# fix.  Its embedded panel is retained in the raw download for provenance but
# is not eligible for promotion.  Wait for the hash-bound corrected evaluator,
# which enforces the Malrule contract on every prompt.
while [[ ! -f "$corrected_c4/complete" ]]; do
  heartbeat
  [[ ! -f "$corrected_c4/needs_attention" ]] || fail "corrected diagnosis C4 needs attention"
  sleep_unlocked 60
done
test -f "$corrected_c4/c4_semantic/summary.json" || fail "corrected diagnosis C4 summary missing"
cp "$corrected_c4/c4_semantic/summary.json" "$state/c4_semantic/summary.json"
cp "$corrected_c4/c4_semantic/metrics.json" "$state/c4_semantic/metrics.json"
cp "$corrected_c4/c4_semantic/predictions_redacted.jsonl" "$state/c4_semantic/predictions_redacted.jsonl"
printf '%s\n' "corrected_evaluator_sha256=660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca" \
  > "$state/c4_semantic/corrected_protocol.txt"

"$python_bin" - "$state/t3_semantic/metrics.json" "$state/c4_semantic/summary.json" "${answer_c4[0]}" \
  "$state/final_semantic_summary.json" <<'PY'
import json,pathlib,sys
t3=json.load(open(sys.argv[1])); c4=json.load(open(sys.argv[2])); answer=json.load(open(sys.argv[3]))
assert t3['judge_backend']=='trapi' and t3['judge_errors']==0,t3
groups=t3['metrics_by_group']; support=sum(int(x['llm_judge_n']) for x in groups.values())
accuracy=sum(float(x['llm_judge_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
out={'stage':'state8_diagnosis_first_matched_semantic_summary_v1','status':'complete',
 't3_semantic_accuracy':accuracy,'t3_support':support,'diagnosis_c4':c4,
 'answer_c4':answer,'answer_c4_pass':bool(answer['summary']['scientific_gate_pass']),
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'canonical_label_role':'auxiliary_supervision_and_reporting',
 'state_token_count':8,'final_inference_route':'strict_state_only'}
pathlib.Path(sys.argv[4]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
while [[ ! -f "$full_teacher/complete" ]]; do
  heartbeat
  [[ ! -f "$full_teacher/needs_attention" ]] || fail "full Teacher evaluation needs attention"
  sleep_unlocked 60
done
[[ -f "$full_teacher/scientific_gate_pass" ]] || fail "full Teacher semantic gate failed"
set +e
"$python_bin" scripts/select_state8_diagnosis_first_matched_curriculum.py \
  --root "$state" --full-teacher-gate "$full_teacher/teacher_full_gate.json" \
  --full-teacher-provenance "$full_teacher/teacher_full_provenance_gate.json" \
  --output "$state/selection.json" > "$state/selection.log" 2>&1
selection_rc=$?
set -e
if (( selection_rc == 0 )); then
  az storage blob upload --auth-mode login --account-name csnewsandfeeds4150361735 \
    --container-name amulet \
    --name studentmodeling/gates/state8_diagnosis_first_matched_v1/selection.json \
    --file "$state/selection.json" --overwrite >> "$state/watcher.log" 2>&1 \
    || fail "matched State8 promotion gate upload failed"
  touch "$state/scientific_gate_pass"
elif (( selection_rc == 2 )); then
  touch "$state/scientific_gate_failed"
else
  fail "matched State8 joint selector crashed rc=$selection_rc"
fi
touch "$state/complete"
heartbeat
log "matched State8 curriculum and full joint promotion gate reached terminal state rc=$selection_rc"
