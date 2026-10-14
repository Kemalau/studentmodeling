#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
fit_dir="$pipeline/task_retention_promoted_final_fit"
state_dir="$pipeline/task_retention_promoted_final_eval"
semantic_dir="$pipeline/task_retention_promoted_t3_semantic_judge"
config="$project_dir/amlt/feeds_layerwise_task_retention_promoted_final_eval_20260802.yaml"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
base_experiment="layerwise-task-retention-promoted-final-eval-seed42-20260802"
active_file="$state_dir/active_experiment.txt"
jobs_file="$state_dir/active_jobs.txt"
preserved="$state_dir/preserved_results"
mkdir -p "$state_dir" "$semantic_dir" "$preserved"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention" "$semantic_dir/needs_attention"

heartbeat(){
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$semantic_dir/controller.heartbeat"
}
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
q=select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']; n=int(sys.argv[2])
assert int(q['standard']['limit'])-int(q['standard']['used'])>=n
assert int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)>=n
PY
    then return 0; fi
    sleep 120
  done
}
preserve_passed(){
  local experiment="$1" status
  status="$(amlt status "$experiment" 2>&1)" || return 1
  grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status" || return 0
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$preserved" >> "$state_dir/controller.log" 2>&1
}
submit(){
  local experiment="$1" description="$2"; shift 2
  local selectors=("$@") names=() item output rc current class
  for item in "${selectors[@]}"; do names+=("${item#:}"); done
  wait_quota "${#selectors[@]}"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python scripts/validate_layerwise_task_retention_promoted_final_eval_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "${names[@]}" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "$description" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${experiment}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$active_file"
  printf '%s\n' "${names[@]}" > "$jobs_file"
}

while [[ ! -f "$fit_dir/scientific_gate_pass" ]]; do
  heartbeat
  [[ ! -f "$fit_dir/needs_attention" ]] || fail "promoted final fit needs attention"
  sleep 120
done

all_jobs=(
  layerwise-promoted-final-eval-t1-t6-seed42
  layerwise-promoted-final-eval-t2-seed42
  layerwise-promoted-final-eval-t3-seed42
  layerwise-promoted-final-eval-t5-seed42
  layerwise-promoted-final-eval-c4-seed13
  layerwise-promoted-final-eval-c4-seed42
  layerwise-promoted-final-eval-c4-seed97
)
if [[ -s "$active_file" ]]; then
  experiment="$(<"$active_file")"
else
  selectors=(); for item in "${all_jobs[@]}"; do selectors+=(":$item"); done
  submit "$base_experiment" \
    "Five-way frozen T1/T2/T3/T5/T6/C4 evaluation of the promoted exact-epoch layerwise model" \
    "${selectors[@]}" || fail "promoted final-eval submission failed"
  experiment="$base_experiment"
fi
expected_jobs="$(wc -l < "$jobs_file")"
wave=0
if [[ "$experiment" =~ recovery([0-9]+)-seed42 ]]; then wave="${BASH_REMATCH[1]}"; fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
    >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|ServiceUnavailable' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$experiment" || fail "could not download promoted evaluation results"
    break
  fi
  preserve_passed "$experiment" || fail "could not preserve passed promoted-eval siblings"
  mapfile -t failed < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal promoted evaluation has no failed jobs"
  (( wave < 5 )) || fail "promoted final evaluation exhausted five recovery waves"
  wave=$((wave+1)); selectors=(); for item in "${failed[@]}"; do selectors+=(":$item"); done
  experiment="layerwise-task-retention-promoted-final-eval-recovery${wave}-seed42-20260802"
  submit "$experiment" "Promoted formal evaluation failed-sibling recovery=$wave jobs=${failed[*]}" \
    "${selectors[@]}" || fail "promoted eval recovery submission failed wave=$wave"
  expected_jobs="${#failed[@]}"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$preserved/." "$state_dir/download/"
readarray -t threshold_audits < <(find "$state_dir/download" -type f -name t6_threshold_application.json)
[[ ${#threshold_audits[@]} -eq 1 ]] || fail "expected one frozen T6 threshold audit"
python - "${threshold_audits[0]}" "$fit_dir/download" <<'PY' \
  || fail "frozen T6 threshold audit is invalid"
import json,pathlib,sys
audit=json.load(open(sys.argv[1],encoding='utf-8'))
receipts=list(pathlib.Path(sys.argv[2]).rglob('promoted_final_fit_receipt.json'))
assert len(receipts)==1,receipts
receipt=json.load(open(receipts[0],encoding='utf-8'))
assert audit['stage']=='layerwise_t6_frozen_threshold_application_v1',audit
assert float(audit['threshold'])==float(receipt['t6_decision_threshold']),audit
assert audit['calibration_artifact_sha256']==receipt['t6_calibration_artifact_sha256'],audit
assert audit['rows_updated']==976,audit
PY
readarray -t predictions < <(find "$state_dir/download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
[[ ${#predictions[@]} -eq 1 ]] || fail "expected one promoted T3 prediction file"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "promoted T3 eval panel drift"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python; backend=responses; extra="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  python_bin=/home/v-yuejiali/fkft/.venv/bin/python; backend=trapi
  extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi
[[ -x "$python_bin" ]] || fail "promoted T3 judge environment unavailable"
mkdir -p "$semantic_dir/full1494"
attempt=0
while [[ ! -f "$semantic_dir/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$semantic_dir/full1494" --judge-backend "$backend" \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$semantic_dir/cache.jsonl" --judge-audit-samples 100 --seed 42 \
      >> "$semantic_dir/full.log" 2>&1 \
    && "$python_bin" scripts/validate_t3_semantic_result.py \
      --metrics "$semantic_dir/full1494/metrics.json" \
      --output "$semantic_dir/t3_semantic_gate.json" >> "$semantic_dir/controller.log" 2>&1; then
    touch "$semantic_dir/scientific_gate_pass" "$semantic_dir/complete"
    break
  fi
  (( attempt < 30 )) || { touch "$semantic_dir/needs_attention"; fail "promoted T3 judge exhausted"; }
  sleep 120
done

python scripts/summarize_layerwise_unified_final_eval.py \
  --root "$state_dir/download" --output "$state_dir/summary.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || { touch "$state_dir/scientific_gate_failed"; fail "promoted final scientific gate failed"; }
# The strict cross-stage audit intentionally requires every stage marker.  Make
# the two final-evaluation markers visible for that audit only after its own
# scientific summary has passed, and roll them back if the cross-stage audit
# rejects any training, artifact, semantic, support, or C4 invariant.
touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
if ! python scripts/validate_layerwise_promoted_unified_pipeline.py \
    --root "$pipeline" --output "$pipeline/overall_completion.json" \
    >> "$state_dir/controller.log" 2>&1; then
  rm -f "$state_dir/scientific_gate_pass" "$state_dir/complete" \
    "$pipeline/overall_completion.json" "$pipeline/overall_complete"
  fail "promoted overall completion audit failed"
fi
touch "$pipeline/overall_complete"
log "promoted unified model formal evaluation and completion audit passed"
