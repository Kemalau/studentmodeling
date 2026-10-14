#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
config="$project/${CONFIG_PATH:-amlt/feeds_external_state_teacher_eedi_full_diagnostic_20260804.yaml}"
validator="$project/${VALIDATOR:-scripts/validate_external_state_teacher_eedi_full_amlt.py}"
base_experiment="${BASE_EXPERIMENT:-external-state-teacher-eedi-full-repair-seed42-20260804}"
state_slug="${STATE_SLUG:-external_state_teacher_eedi_full_20260804}"
state_dir="$project/results/layerwise_unified_final_pipeline_20260731/$state_slug"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
job_prefix="${JOB_PREFIX:-external-state-teacher}"
recovery_prefix="${RECOVERY_PREFIX:-external-state-teacher}"
jobs=("${job_prefix}-cw025-seed42" "${job_prefix}-cw100-seed42")

mkdir -p "$state_dir/preserved" "$state_dir/passed_receipts" "$state_dir/t3_semantic"
cd "$project"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state_dir/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if "$python_bin" - "$quota" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
q=select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']
assert int(q['standard']['limit'])-int(q['standard']['used']) >= 4
assert int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0) >= 4
PY
    then return 0; fi
    sleep 120
  done
}

preserve_passed(){
  local experiment="$1"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state_dir/preserved" >> "$state_dir/controller.log" 2>&1
}

declare -A arm_experiment waves
arm_map="$state_dir/arm_experiments.tsv"
if [[ -s "$arm_map" ]]; then
  while IFS=$'\t' read -r job experiment wave; do
    [[ -n "$job" && -n "$experiment" ]] || continue
    arm_experiment["$job"]="$experiment"; waves["$job"]="${wave:-0}"
  done < "$arm_map"
else
  for job in "${jobs[@]}"; do
    arm_experiment["$job"]="$base_experiment"; waves["$job"]=0
  done
fi

save_arm_map(){
  local temporary="$arm_map.partial.$$" job
  : > "$temporary"
  for job in "${jobs[@]}"; do
    printf '%s\t%s\t%s\n' "$job" "${arm_experiment[$job]}" "${waves[$job]}" >> "$temporary"
  done
  mv "$temporary" "$arm_map"
}
save_arm_map

submit_recovery(){
  local job="$1" wave="$2" experiment current class output rc
  experiment="${recovery_prefix}-${job#${job_prefix}-}-recovery${wave}-20260804"
  wait_quota
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$validator" --config "$state_dir/${experiment}.preflight.dump.yaml" \
    --jobs "$job" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "External-state-teacher infrastructure recovery=$wave arm=$job; objective and frozen artifacts unchanged" 2>&1)"; rc=$?
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
  arm_experiment["$job"]="$experiment"
  save_arm_map
}

while true; do
  heartbeat
  transient=false
  for job in "${jobs[@]}"; do
    receipt="$state_dir/passed_receipts/$job.complete"
    [[ -f "$receipt" ]] && continue
    experiment="${arm_experiment[$job]}"
    status="$(timeout 90 amlt status "$experiment" ":$job" 2>&1)" || { transient=true; break; }
    printf '[%s] experiment=%s job=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$job" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|InternalServerError|ServiceUnavailable' <<< "$status"; then
      transient=true; break
    fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
    then
      preserve_passed "$experiment" || { transient=true; break; }
      printf 'experiment=%s\njob=%s\npassed_at=%s\n' "$experiment" "$job" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$receipt"
      log "recorded Passed arm=$job experiment=$experiment"
      continue
    fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_active_job
raise SystemExit(0 if has_active_job(sys.argv[1]) else 1)
PY
    then continue; fi
    mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
    )
    [[ ${#failed[@]} -eq 1 && "${failed[0]}" == "$job" ]] \
      || fail "unrecognized terminal state experiment=$experiment arm=$job"
    timeout 180 amlt logs view -n 1500 "$experiment" ":$job" \
      > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
    waves["$job"]=$((waves["$job"]+1))
    (( waves["$job"] <= 6 )) || fail "arm exhausted six recovery waves: $job"
    preserve_passed "$experiment" || true
    submit_recovery "$job" "${waves[$job]}" \
      || fail "recovery submission failed arm=$job wave=${waves[$job]}"
    log "recovered arm=$job wave=${waves[$job]} experiment=${arm_experiment[$job]}"
  done
  [[ "$transient" == true ]] && { sleep 120; continue; }
  all_passed=true
  for job in "${jobs[@]}"; do
    [[ -f "$state_dir/passed_receipts/$job.complete" ]] || all_passed=false
  done
  [[ "$all_passed" == true ]] && break
  sleep 120
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved/." "$state_dir/download/"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "frozen formal evaluation panel drift"

(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
summary_args=()
for job in "${jobs[@]}"; do
  mapfile -t predictions < <(find "$state_dir/download" -type f \
    -path "*${job}*/eval/formal_screen/predictions.jsonl" | sort)
  [[ ${#predictions[@]} -eq 1 ]] \
    || fail "expected one formal prediction file arm=$job found=${#predictions[@]}"
  prediction="${predictions[0]}"
  job_root="$(dirname "$(dirname "$(dirname "$prediction")")")"
  slug="${job#external-state-teacher-}"
  output="$state_dir/t3_semantic/$slug"; mkdir -p "$output"
  attempt=0
  while [[ ! -f "$output/complete" ]]; do
    attempt=$((attempt+1)); heartbeat
    if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" \
        --output-dir "$output" --judge-backend trapi --judge-workers 12 \
        --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
        --judge-audit-samples 75 --seed 42 >> "$output/judge.log" 2>&1; then
      touch "$output/complete"
      break
    fi
    (( attempt < 30 )) || fail "T3 semantic judge exhausted arm=$job"
    sleep 120
  done
  summary_args+=(--job "$job_root" --t3-metrics "$output/metrics.json")
done

"$python_bin" scripts/summarize_external_state_teacher_diagnostic.py \
  "${summary_args[@]}" --output "$state_dir/selection.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "external-state-teacher selection failed"
if "$python_bin" - "$state_dir/selection.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1],encoding='utf-8'))['any_all_gates_pass'] else 1)
PY
then touch "$state_dir/scientific_gate_pass"; else rm -f "$state_dir/scientific_gate_pass"; fi
touch "$state_dir/complete"
heartbeat
log "external-state-teacher diagnostic complete"
