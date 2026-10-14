#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/c3_exposure_matched_round1500_hashfix_v2"
config="$project_dir/amlt/feeds_c3_exposure_matched_round1500_20260804.yaml"
base_experiment=c3-exposure-matched-round1500-seed20260804
subscription=b6dc87f3-c479-49c8-8cb5-7896da3ff895
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"
jobs=(
  c3-matched-malrule-correctness-seed20260804
  c3-matched-malrule-response-seed20260804
  c3-matched-malrule-misconception-seed20260804
)

mkdir -p "$state_dir/preserved" "$state_dir/passed" "$state_dir/t3_semantic"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention" \
  "$state_dir/scientific_gate_pass" "$state_dir/scientific_gate_failed"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

# A dedicated Azure CLI cache is useful only while its refresh token remains
# valid.  Treating an expired cache as a transient network error left the C3
# controller heartbeating for days without observing or recovering any job.
# Probe the inherited cache first, then the two project-approved caches, and
# fail visibly if none can authenticate.
select_azure_profile(){
  local candidate
  if [[ -n "${AZURE_CONFIG_DIR:-}" ]] && \
     AZURE_CONFIG_DIR="$AZURE_CONFIG_DIR" az account get-access-token \
       --subscription "$subscription" --query expiresOn -o tsv >/dev/null 2>&1; then
    rm -f "$state_dir/auth_required"
    return 0
  fi
  for candidate in /home/v-yuejiali/.azure /home/v-yuejiali/.azure-studentmodeling-feeds; do
    if AZURE_CONFIG_DIR="$candidate" az account get-access-token \
         --subscription "$subscription" --query expiresOn -o tsv >/dev/null 2>&1; then
      export AZURE_CONFIG_DIR="$candidate"
      rm -f "$state_dir/auth_required"
      return 0
    fi
  done
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/auth_required"
  return 1
}

select_azure_profile || fail "no approved Azure CLI profile can authenticate"

declare -A experiments waves
map="$state_dir/experiments.tsv"
if [[ -s "$map" ]]; then
  while IFS=$'\t' read -r job experiment wave; do
    experiments["$job"]="$experiment"; waves["$job"]="$wave"
  done < "$map"
else
  for job in "${jobs[@]}"; do
    experiments["$job"]="$base_experiment"; waves["$job"]=0
  done
fi
save_map(){
  local temporary="$map.partial.$$" job
  : > "$temporary"
  for job in "${jobs[@]}"; do
    printf '%s\t%s\t%s\n' "$job" "${experiments[$job]}" "${waves[$job]}" >> "$temporary"
  done
  mv "$temporary" "$map"
}
save_map

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || {
      if grep -Eqi 'No authentication method succeeded|refreshing your credentials' <<< "$quota"; then
        select_azure_profile || fail "Azure authentication expired while waiting for C3 quota"
      fi
      sleep 120
      continue
    }
    if "$python_bin" - "$quota" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
q=select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']
assert int(q['standard']['limit'])-int(q['standard']['used'])>=4
assert int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)>=4
PY
    then return 0; fi
    sleep 120
  done
}

preserve_passed(){
  local experiment="$1"
  timeout 600 amlt results download "$experiment" --status pass \
    --include '*.json' --include 'predictions.jsonl' \
    --include 'predictions_redacted.jsonl' --output "$state_dir/preserved" \
    >> "$state_dir/controller.log" 2>&1
}

submit_recovery(){
  local job="$1" wave="$2" experiment output current class rc
  experiment="c3-hashfix-v2-round1500-${job#c3-matched-}-recovery${wave}-20260805"
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_c3_exposure_matched_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "$job" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "C3 exposure-matched failed-sibling recovery wave=$wave job=$job; frozen one-factor protocol unchanged" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${experiment}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  experiments["$job"]="$experiment"; save_map
}

while true; do
  heartbeat; all_pass=true
  for job in "${jobs[@]}"; do
    [[ -f "$state_dir/passed/$job" ]] && continue
    all_pass=false; experiment="${experiments[$job]}"
    status="$(timeout 90 amlt status "$experiment" ":$job" 2>&1)" || { sleep 60; continue 2; }
    printf '[%s] experiment=%s job=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$job" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials' <<< "$status"; then
      select_azure_profile || fail "Azure authentication expired while monitoring C3 jobs"
      sleep 5; continue 2
    fi
    if grep -Eqi 'connection|timed out|permission|TooManyRequests|ServiceUnavailable' <<< "$status"; then
      sleep 60; continue 2
    fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
    then
      preserve_passed "$experiment" || { sleep 60; continue 2; }
      printf 'experiment=%s\npassed_at=%s\n' "$experiment" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$state_dir/passed/$job"
      log "preserved passed C3 arm=$job experiment=$experiment"
      continue
    fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_active_job
raise SystemExit(0 if has_active_job(sys.argv[1]) else 1)
PY
    then continue; fi
    timeout 180 amlt logs view -n 2000 "$experiment" ":$job" \
      > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
    waves["$job"]=$((waves["$job"]+1))
    (( waves["$job"] <= 6 )) || fail "C3 recovery waves exhausted arm=$job"
    preserve_passed "$experiment" || true
    submit_recovery "$job" "${waves[$job]}" || fail "C3 recovery submit failed arm=$job"
  done
  [[ "$all_pass" == true ]] && break
  sleep 60
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved/." "$state_dir/download/"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "C3 eval manifest drift"

declare -A arm_by_job=(
  [c3-matched-malrule-correctness-seed20260804]=malrule_correctness
  [c3-matched-malrule-response-seed20260804]=malrule_response
  [c3-matched-malrule-misconception-seed20260804]=malrule_misconception
)
for job in "${jobs[@]}"; do
  arm="${arm_by_job[$job]}"
  mapfile -t predictions < <(find "$state_dir/download" -type f \
    -path "*${job}*/eval/formal_screen/predictions.jsonl" | sort)
  [[ ${#predictions[@]} -eq 1 ]] \
    || fail "expected one C3 T3 prediction file arm=$arm found=${#predictions[@]}"
  output="$state_dir/t3_semantic/$arm"; mkdir -p "$output"; heartbeat
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$output/judged" --judge-backend trapi --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state_dir/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
    >> "$output/live.log" 2>&1 || fail "C3 T3 semantic judge failed arm=$arm"
  "$python_bin" scripts/validate_checkpoint_t3_semantic_result.py \
    --metrics "$output/judged/metrics.json" --output "$output/gate.json" \
    --checkpoint-round 1750 >> "$output/live.log" 2>&1; rc=$?
  (( rc == 0 || rc == 2 )) || fail "C3 semantic validation error arm=$arm rc=$rc"
done

"$python_bin" scripts/summarize_c3_exposure_matched.py \
  --root "$state_dir/download" --t3-root "$state_dir/t3_semantic" \
  --output "$state_dir/summary.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "C3 controlled summary failed"
if "$python_bin" - "$state_dir/summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['scientific_gate_pass'] else 1)
PY
then touch "$state_dir/scientific_gate_pass"; else touch "$state_dir/scientific_gate_failed"; fi
touch "$state_dir/complete"; heartbeat
log "C3 exposure-matched experiment complete; hypothesis and completion reported separately"
