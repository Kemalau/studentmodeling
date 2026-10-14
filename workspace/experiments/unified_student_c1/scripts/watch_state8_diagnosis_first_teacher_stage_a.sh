#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_diagnosis_first_teacher_stage_a_20260810"
config=amlt/feeds_state8_diagnosis_first_teacher_stage_a_20260810.yaml
experiment=state8-diagnosis-first-teacher-stage-a-20260810
recovery_prefix=state8-diagnosis-first-teacher-stage-a-recovery
job=state8-diagnosis-first-full-history-teacher-seed66
max_recovery_waves=12
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
account=csnewsandfeeds4150361735
container=amulet
prefix=studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"; flock -n 9 || exit 0
echo $$ > "$state/watcher.active.pid"
trap 'rm -f "$state/watcher.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/watcher.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }

if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
fi
wave=0
if [[ "$experiment" =~ recovery([0-9]+)-20260810$ ]]; then
  wave="${BASH_REMATCH[1]}"
fi

submit_recovery(){
  local next="$1" output rc class
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_diagnosis_first_teacher_stage_a_amlt.py \
    --config "$config" >> "$state/watcher.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" --dump -t Feeds \
    > "$state/$next.preflight.yaml" 2> "$state/$next.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_diagnosis_first_teacher_stage_a_amlt.py \
    --config "$state/$next.preflight.yaml" >> "$state/watcher.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" "$next" -t Feeds -y \
    --description "Automatic resumable recovery of schema-matched full-history diagnosis teacher" 2>&1)"
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

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep_unlocked 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
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
    || fail "teacher exhausted automatic recovery waves experiment=$experiment"
  wave=$((wave+1))
  next="${recovery_prefix}${wave}-20260810"
  until submit_recovery "$next"; do
    heartbeat
    log "transient teacher recovery submission failure wave=$wave; retrying"
    sleep_unlocked 120
  done
  experiment="$next"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted teacher recovery wave=$wave experiment=$experiment"
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass \
  --include 'adapter_model.safetensors' --include 'adapter_config.json' \
  --include 'predictions.jsonl' --include 'metrics.json' \
  --include 'teacher_eval.jsonl' --include 'teacher_view_receipt.json' \
  --include 'publish_receipt.json' --output "$state/download" \
  >> "$state/watcher.log" 2>&1 || fail "teacher result download failed"
mapfile -t predictions < <(find "$state/download" -type f -path '*/eval/t3/predictions.jsonl')
mapfile -t evaluations < <(find "$state/download" -type f -path '*/metrics/teacher_eval.jsonl')
mapfile -t receipts < <(find "$state/download" -type f -path '*/metrics/publish_receipt.json')
[[ ${#predictions[@]} -eq 1 && ${#evaluations[@]} -eq 1 && ${#receipts[@]} -eq 1 ]] \
  || fail "teacher artifact cardinality drift"
mkdir -p "$state/semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${predictions[0]}" --eval-jsonl "${evaluations[0]}" \
  --output-dir "$state/semantic" --judge-backend trapi \
  --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
  --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/semantic/cache.jsonl" --judge-audit-samples 75 --seed 66 \
  >> "$state/semantic.log" 2>&1 || fail "TRAPI semantic judge failed"
"$python_bin" scripts/validate_state8_diagnosis_first_teacher_screen.py \
  --metrics "$state/semantic/metrics.json" --output "$state/teacher_gate.json" \
  >> "$state/watcher.log" 2>&1
gate_rc=$?
if (( gate_rc != 0 )); then
  touch "$state/scientific_gate_failed" "$state/needs_attention"
  log "teacher semantic screen failed; downstream State8 curriculum intentionally stopped"
  exit "$gate_rc"
fi
"$python_bin" - "${receipts[0]}" "$state/teacher_gate.json" \
  "$state/teacher_provenance_gate.json" "$state/completed_publish_receipt.json" <<'PY'
import json,pathlib,sys
receipt=json.load(open(sys.argv[1])); gate=json.load(open(sys.argv[2]))
assert receipt['stage']=='state8_diagnosis_first_full_history_teacher_publish_v1',receipt
assert receipt['diagnosis_field_order']==['diagnosis','evidence','label'],receipt
assert gate['pass'] is True and gate['semantic_judge_authoritative'] is True,gate
value={'status':'complete','stage':'state8_diagnosis_first_teacher_provenance_gate_v1',
 'teacher_gate_pass':True,'teacher_adapter_sha256':receipt['adapter_sha256'],
 'evaluation_input_sha256':receipt['eval_sha256'],
 'diagnosis_field_order':['diagnosis','evidence','label'],
 'output_schema':'diagnosis_evidence_label_v1',
 'canonical_label_role':'auxiliary_supervision_and_reporting',
 'teacher_history_visible':True,'student_history_visible_at_final_inference':False,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'screen_rows':75}
pathlib.Path(sys.argv[3]).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
completed={**receipt,'status':'complete','semantic_teacher_gate_pass':True,
 'teacher_gate_sha256':__import__('hashlib').sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest(),
 'teacher_provenance_gate_sha256':__import__('hashlib').sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()}
pathlib.Path(sys.argv[4]).write_text(json.dumps(completed,indent=2,sort_keys=True)+'\n')
PY
az storage blob upload --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$prefix/metrics/teacher_gate.json" \
  --file "$state/teacher_gate.json" --overwrite \
  >> "$state/watcher.log" 2>&1 || fail "teacher gate upload failed"
az storage blob upload --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$prefix/metrics/teacher_provenance_gate.json" \
  --file "$state/teacher_provenance_gate.json" --overwrite \
  >> "$state/watcher.log" 2>&1 || fail "teacher provenance upload failed"
az storage blob upload --auth-mode login --account-name "$account" \
  --container-name "$container" --name "$prefix/metrics/publish_receipt.json" \
  --file "$state/completed_publish_receipt.json" --overwrite \
  >> "$state/watcher.log" 2>&1 || fail "completed teacher receipt upload failed"
touch "$state/scientific_gate_pass" "$state/complete"
heartbeat
log "diagnosis-first teacher passed semantic screen and is ready for State8 curriculum"

# The user explicitly authorized the full gated curriculum.  Submission is
# conditional on this semantic gate, revalidated from a rendered AMLT config,
# and serialized with every other Feeds submission in this project.
curriculum_config=amlt/feeds_state8_diagnosis_first_matched_curriculum_20260810.yaml
curriculum_experiment=state8-diagnosis-first-matched-curriculum-20260810
"$python_bin" scripts/validate_state8_diagnosis_first_matched_curriculum_amlt.py \
  --config "$curriculum_config" >> "$state/watcher.log" 2>&1 \
  || fail "matched State8 source config invalid"
exec 8>"$project/results/feeds_submission_global.lock"; flock 8
amlt run "$curriculum_config" --dump -t Feeds \
  > "$state/curriculum.preflight.yaml" 2> "$state/curriculum.preflight.stderr" \
  || { flock -u 8; fail "matched State8 preflight render failed"; }
"$python_bin" scripts/validate_state8_diagnosis_first_matched_curriculum_amlt.py \
  --config "$state/curriculum.preflight.yaml" >> "$state/watcher.log" 2>&1 \
  || { flock -u 8; fail "matched State8 rendered config invalid"; }
current="$(amlt status --no-update-db "$curriculum_experiment" 2>&1)"; rc=$?
class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "matched State8 experiment lookup ambiguous"; }
if [[ "$class" == missing ]]; then
  amlt --json-tables ti sing -t Feeds > "$state/curriculum.quota.pre_submit.json" \
    || { flock -u 8; fail "matched State8 quota refresh failed"; }
  amlt run "$curriculum_config" "$curriculum_experiment" -t Feeds -y \
    --description "Semantic-gated schema-matched teacher to 8 State tokens; five dropout phases; strict state-only inference; low-dose C4" \
    > "$state/curriculum.submission.txt" 2>&1 \
    || { flock -u 8; fail "matched State8 submission failed"; }
elif [[ "$class" != exists ]]; then
  flock -u 8; fail "matched State8 experiment lookup ambiguous"
fi
flock -u 8
printf '%s\n' "$curriculum_experiment" > "$state/active_curriculum_experiment.txt"
touch "$state/curriculum_submitted"
log "submitted gated matched-teacher State8 curriculum experiment=$curriculum_experiment"
