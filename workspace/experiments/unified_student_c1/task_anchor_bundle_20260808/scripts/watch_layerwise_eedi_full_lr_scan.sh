#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_slug="${STATE_SLUG:-eedi_full_lr_scan}"
state_dir="$pipeline/$state_slug"
config="$project_dir/${CONFIG_PATH:-amlt/feeds_layerwise_eedi_full_lr_scan_20260802.yaml}"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
train_jsonl="$project_dir/data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl"
loss_profile="${LOSS_PROFILE_EXPECTED:-state_path}"
summary_branch="${SUMMARY_BRANCH_EXPECTED:-eedi_full}"
validator="${VALIDATOR:-scripts/validate_layerwise_eedi_full_lr_scan_amlt.py}"
base_experiment="${BASE_EXPERIMENT:-layerwise-eedi-full-statepath-lr-scan}"
job_prefix="${JOB_PREFIX:-layerwise-eedi-full-statepath}"
experiment_description="${EXPERIMENT_DESCRIPTION:-Full-support Eedi $loss_profile formation LR scan}"
experiment_date="${EXPERIMENT_DATE:-20260802}"
mkdir -p "$state_dir/preserved" "$state_dir/t3_semantic"
mkdir -p "$state_dir/passed_receipts"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

jobs=("${job_prefix}-lr2e6-seed42" "${job_prefix}-lr5e6-seed42")

wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$required" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
q=select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']; n=int(sys.argv[2])
assert int(q['standard']['limit'])-int(q['standard']['used'])>=n
assert int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)>=n
PY
    then return 0; fi
    sleep 120
  done
}

preserve_passed(){
  local experiment="$1"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include 'predictions.jsonl' --include 'predictions_redacted.jsonl' \
    --output "$state_dir/preserved" >> "$state_dir/controller.log" 2>&1
}

submit_wave(){
  local wave="$1"; shift
  local selected=("$@") selectors=() experiment output rc current class
  for job in "${selected[@]}"; do selectors+=(":$job"); done
  experiment="${base_experiment}-recovery${wave}-seed42-${experiment_date}"
  wait_quota "$((4 * ${#selected[@]}))"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python "$validator" \
    --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "${selected[@]}" \
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
      --description "$experiment_description recovery=$wave jobs=${selected[*]}; checkpoint/data/C4 frozen" 2>&1)"; rc=$?
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
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
  printf '%s\n' "${selected[@]}" > "$state_dir/active_jobs.txt"
  submitted_experiment="$experiment"
}

arm_map="$state_dir/arm_experiments.tsv"
declare -A arm_experiment=()

save_arm_map(){
  local tmp="$arm_map.tmp.$$" job
  : > "$tmp"
  for job in "${jobs[@]}"; do
    printf '%s\t%s\n' "$job" "${arm_experiment[$job]}" >> "$tmp"
  done
  mv "$tmp" "$arm_map"
}

wave=0
if [[ -s "$arm_map" ]]; then
  while IFS=$'\t' read -r job experiment; do
    [[ -n "$job" && -n "$experiment" ]] || continue
    arm_experiment["$job"]="$experiment"
    if [[ "$experiment" =~ recovery([0-9]+)-seed42 ]]; then
      (( BASH_REMATCH[1] > wave )) && wave="${BASH_REMATCH[1]}"
    fi
  done < "$arm_map"
elif [[ -s "$state_dir/active_experiment.txt" && -s "$state_dir/active_jobs.txt" ]]; then
  # Migrate the original aggregate controller state without resubmitting it.
  experiment="$(<"$state_dir/active_experiment.txt")"
  while IFS= read -r job; do
    [[ -n "$job" ]] && arm_experiment["$job"]="$experiment"
  done < "$state_dir/active_jobs.txt"
  [[ "$experiment" =~ recovery([0-9]+)-seed42 ]] && wave="${BASH_REMATCH[1]}"
else
  wave=0
  submit_wave "$wave" "${jobs[@]}" || fail "initial LR-scan submission failed"
  for job in "${jobs[@]}"; do arm_experiment["$job"]="$submitted_experiment"; done
fi

for job in "${jobs[@]}"; do
  [[ -n "${arm_experiment[$job]:-}" ]] || fail "missing experiment mapping for arm=$job"
done
save_arm_map

while true; do
  heartbeat
  all_passed=true
  transient=false
  for job in "${jobs[@]}"; do
    receipt="$state_dir/passed_receipts/$job.complete"
    [[ -f "$receipt" ]] && continue
    all_passed=false
    if [[ -f "$state_dir/deferred_jobs.txt" ]] \
        && grep -Fxq "$job" "$state_dir/deferred_jobs.txt"; then
      continue
    fi
    experiment="${arm_experiment[$job]}"
    status="$(timeout 90 amlt status "$experiment" ":$job" 2>&1)" || { transient=true; break; }
    printf '[%s] experiment=%s job=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$job" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|InternalServerError|ServiceUnavailable' <<< "$status"; then
      transient=true
      break
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
    then
      preserve_passed "$experiment" || { transient=true; break; }
      printf 'experiment=%s\njob=%s\npassed_at=%s\n' "$experiment" "$job" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$receipt"
      continue
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_active_job
raise SystemExit(0 if has_active_job(sys.argv[1]) else 1)
PY
    then continue; fi
    mapfile -t failed < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
    )
    [[ ${#failed[@]} -eq 1 && "${failed[0]}" == "$job" ]] \
      || fail "unrecognized terminal state experiment=$experiment arm=$job"
    (( wave < 6 )) || fail "LR scan exhausted six recovery waves"
    wave=$((wave+1))
    preserve_passed "$experiment" || true
    submit_wave "$wave" "$job" || fail "LR-scan recovery submission failed wave=$wave arm=$job"
    arm_experiment["$job"]="$submitted_experiment"
    save_arm_map
    log "recovered terminal arm=$job previous=$experiment current=$submitted_experiment"
  done
  [[ "$transient" == true ]] && { sleep 120; continue; }
  if [[ "$all_passed" == true ]]; then break; fi
  # A pass may have been recorded during this iteration. Re-evaluate receipts
  # immediately so the controller does not add an unnecessary polling delay.
  all_passed=true
  for job in "${jobs[@]}"; do
    [[ -f "$state_dir/passed_receipts/$job.complete" ]] || all_passed=false
  done
  [[ "$all_passed" == true ]] && break
  sleep 120
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved/." "$state_dir/download/" 2>/dev/null || true
readarray -t predictions < <(find "$state_dir/download" -type f -path '*/eval/formal_screen/predictions.jsonl' | sort)
[[ ${#predictions[@]} -eq 2 ]] || fail "expected two LR-scan formal prediction files"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python; backend=responses; extra="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  python_bin=/home/v-yuejiali/fkft/.venv/bin/python; backend=trapi; extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi
summaries=()
for prediction in "${predictions[@]}"; do
  if [[ "$prediction" == *lr2e6* ]]; then slug=lr2e6; lr=2e-6; else slug=lr5e6; lr=5e-6; fi
  output="$state_dir/t3_semantic/$slug"; mkdir -p "$output"; heartbeat
  PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" \
    --output-dir "$output" --judge-backend "$backend" --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state_dir/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
    >> "$state_dir/t3_semantic/$slug.log" 2>&1 || fail "LR-scan T3 judge failed slug=$slug"
  python scripts/summarize_t3_seen_unseen.py \
    --train-jsonl "$train_jsonl" --eval-jsonl "$eval_jsonl" \
    --predictions-jsonl "$prediction" \
    --judgments-jsonl "$output/predictions_redacted.jsonl" \
    --output "$output/seen_unseen.json" \
    >> "$state_dir/t3_semantic/$slug.log" 2>&1 \
    || fail "LR-scan T3 seen/unseen diagnostic failed slug=$slug"
  python scripts/summarize_layerwise_single_task_retention.py \
    --root "$state_dir/download" --t3-metrics "$output/metrics.json" \
    --branch "$summary_branch" --loss-profile "$loss_profile" --learning-rate "$lr" \
    --output "$state_dir/$slug.summary.json" >> "$state_dir/controller.log" 2>&1 \
    || fail "LR-scan summary failed slug=$slug"
  summaries+=("$state_dir/$slug.summary.json")
done
python scripts/summarize_layerwise_eedi_full_lr_scan.py --summaries "${summaries[@]}" \
  --loss-profile "$loss_profile" --branch "$summary_branch" \
  --output "$state_dir/comparison.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "LR-scan comparison failed"
touch "$state_dir/complete"; heartbeat; log "full-support Eedi LR scan complete"
