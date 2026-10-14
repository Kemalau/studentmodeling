#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_state_path_ab"
config="$project_dir/amlt/feeds_layerwise_eedi_state_path_ab_20260802.yaml"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
base_experiment="layerwise-eedi-state-path-ab-seed42-20260802"
mkdir -p "$state_dir/preserved" "$state_dir/t3_semantic"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

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
    --output "$state_dir/preserved" >> "$state_dir/controller.log" 2>&1 || true
}

submit_recovery(){
  local wave="$1"; shift
  local jobs=("$@") selectors=() experiment output rc current class
  for job in "${jobs[@]}"; do selectors+=(":$job"); done
  experiment="layerwise-eedi-state-path-ab-recovery${wave}-seed42-20260802"
  wait_quota "$((4 * ${#jobs[@]}))"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python scripts/validate_layerwise_eedi_state_path_ab_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "${jobs[@]}" \
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
      --description "Eedi-dense objective A/B failed-sibling recovery=$wave jobs=${jobs[*]}; scientific recipe unchanged" 2>&1)"; rc=$?
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
  printf '%s\n' "${jobs[@]}" > "$state_dir/active_jobs.txt"
}

experiment="$(<"$state_dir/active_experiment.txt")"
if [[ -s "$state_dir/active_jobs.txt" ]]; then
  expected_jobs="$(wc -l < "$state_dir/active_jobs.txt")"
else
  printf '%s\n' layerwise-eedi-dense-standard-seed42 layerwise-eedi-dense-state-path-seed42 \
    > "$state_dir/active_jobs.txt"
  expected_jobs=2
fi
wave=0
if [[ "$experiment" =~ recovery([0-9]+)-seed42 ]]; then wave="${BASH_REMATCH[1]}"; fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
    >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|InternalServerError|ServiceUnavailable' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$experiment"
    break
  fi
  preserve_passed "$experiment"
  mapfile -t failed < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal Eedi A/B state has no failed sibling"
  (( wave < 6 )) || fail "Eedi A/B exhausted six recovery waves"
  wave=$((wave+1))
  submit_recovery "$wave" "${failed[@]}" || fail "Eedi A/B recovery submission failed wave=$wave"
  experiment="$(<"$state_dir/active_experiment.txt")"
  expected_jobs="${#failed[@]}"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved/." "$state_dir/download/" 2>/dev/null || true
[[ "$(find "$state_dir/download" -type f -path '*/eval/c4/recurrent_causal_gate.json' | wc -l)" -eq 2 ]] \
  || fail "expected two frozen C4 results"
[[ "$(find "$state_dir/download" -type f -path '*/eval/t6_full/metrics.json' | wc -l)" -eq 2 ]] \
  || fail "expected two full-support T6 results"
readarray -t predictions < <(find "$state_dir/download" -type f -path '*/eval/formal_screen/predictions.jsonl' | sort)
[[ ${#predictions[@]} -eq 2 ]] || fail "expected two Eedi A/B formal prediction files"

python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python; backend=responses; extra="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  python_bin=/home/v-yuejiali/fkft/.venv/bin/python; backend=trapi
  extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi
for prediction in "${predictions[@]}"; do
  # Match the concrete job directory.  The experiment directory also contains
  # "state-path-ab", so a broad substring match labels both arms state_path.
  case "$prediction" in
    */layerwise-eedi-dense-state-path-seed42/*) profile=state_path ;;
    */layerwise-eedi-dense-standard-seed42/*) profile=standard ;;
    *) fail "could not identify Eedi A/B profile for prediction=$prediction" ;;
  esac
  output="$state_dir/t3_semantic/$profile"; mkdir -p "$output"
  heartbeat
  PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" \
    --output-dir "$output" --judge-backend "$backend" --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
    --judge-audit-samples 75 --seed 42 >> "$state_dir/t3_semantic/$profile.log" 2>&1 \
    || fail "Eedi A/B T3 judge failed profile=$profile"
done

python scripts/summarize_layerwise_eedi_state_path_ab.py \
  --root "$state_dir/download" --t3-root "$state_dir/t3_semantic" \
  --output "$state_dir/comparison.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "Eedi A/B comparison failed"
touch "$state_dir/complete"
log "Eedi-dense objective A/B comparison complete"
