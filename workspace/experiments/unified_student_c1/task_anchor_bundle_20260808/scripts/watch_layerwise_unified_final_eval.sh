#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
fit_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/final_fit"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/final_eval"
config="$project_dir/amlt/feeds_layerwise_unified_final_eval_20260731.yaml"
base_experiment="layerwise-unified-final-eval-seed42-20260731"
active_file="$state_dir/active_experiment.txt"
expected_jobs_file="$state_dir/expected_jobs.txt"
results_root="$state_dir/preserved_results"
if [[ -s "$active_file" ]]; then active_experiment="$(<"$active_file")"; else active_experiment="$base_experiment"; fi
[[ "$active_experiment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
printf '%s\n' "$active_experiment" > "$active_file"
expected_jobs=5
recovery_wave=0
if [[ "$active_experiment" =~ recovery([0-9]+)-seed42 ]]; then
  recovery_wave="${BASH_REMATCH[1]}"
  if [[ -s "$expected_jobs_file" ]]; then expected_jobs="$(<"$expected_jobs_file")"; else expected_jobs=1; fi
fi
[[ "$expected_jobs" =~ ^[1-5]$ ]] || exit 2
printf '%s\n' "$expected_jobs" > "$expected_jobs_file"
mkdir -p "$state_dir" "$results_root"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

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
    --output "$results_root" >> "$state_dir/controller.log" 2>&1
}

# Upstream recovery is owned by the final-fit controller.  Do not poison this
# stage with a stale marker while the selected exact-epoch fit is recovering.
while [[ ! -f "$fit_dir/scientific_gate_pass" ]]; do heartbeat; sleep 120; done
# Authentication affects only local orchestration.  A token expiry between the
# upstream completion marker and final-eval submission must pause here rather
# than permanently poisoning the scientific evaluation stage.
while ! az account get-access-token \
    --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
    --query expiresOn -o tsv >/dev/null 2>&1; do
  heartbeat
  sleep 120
done
while ! az account set \
    --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 >/dev/null 2>&1; do
  heartbeat
  sleep 120
done

if [[ "$active_experiment" == "$base_experiment" && ! -f "$state_dir/submission.txt" ]]; then
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds > "$state_dir/preflight.dump.yaml" \
    2> "$state_dir/preflight.dump.stderr" || { flock -u 8; fail "eval dump failed"; }
  python scripts/validate_layerwise_unified_final_eval_amlt.py \
    --config "$state_dir/preflight.dump.yaml" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "eval preflight failed"; }
  current="$(amlt status --no-update-db "$active_experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "eval status ambiguous"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$active_experiment" -t Feeds -y \
      --description 'Formal T1 T2 T3 T5 T6 and frozen C4 evaluation of exact-epoch layerwise unified model' 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" > "$state_dir/submission.txt"
    ((rc==0)) || { flock -u 8; fail "eval submission failed"; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" > "$state_dir/submission.txt"
  else
    flock -u 8; fail "eval state ambiguous"
  fi
  flock -u 8
fi

while true; do
  heartbeat
  status="$(amlt status "$active_experiment" 2>&1)"
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$active_experiment" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if python - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$active_experiment" || fail "could not download passed evaluation results"
    break
  fi
  preserve_passed "$active_experiment" || fail "could not preserve passed evaluation siblings"
  printf '%s\n' "$status" > "$state_dir/${active_experiment}.terminal_status.txt"
  amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$active_experiment" > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal evaluation state has no recoverable jobs"
  (( recovery_wave < 3 )) || fail "formal evaluation exhausted three recovery waves"
  recovery_wave=$((recovery_wave+1)); expected_jobs="${#failed_jobs[@]}"
  printf '%s\n' "$expected_jobs" > "$expected_jobs_file"
  wait_quota "$expected_jobs"
  next_experiment="layerwise-unified-final-eval-recovery${recovery_wave}-seed42-20260801"
  selectors=(); for job in "${failed_jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$state_dir/${next_experiment}.preflight.dump.yaml" \
    2> "$state_dir/${next_experiment}.preflight.stderr" || { flock -u 8; fail "eval recovery dump failed"; }
  python scripts/validate_layerwise_unified_final_eval_amlt.py \
    --config "$state_dir/${next_experiment}.preflight.dump.yaml" --jobs "${failed_jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; fail "eval recovery preflight failed"; }
  current="$(amlt status --no-update-db "$next_experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "eval recovery status ambiguous"; }
  if [[ "$class" == missing ]]; then
    amlt run "$config" "${selectors[@]}" "$next_experiment" -t Feeds -y \
      --description "Formal layerwise unified evaluation recovery=${recovery_wave}; jobs=${failed_jobs[*]}" \
      > "$state_dir/${next_experiment}.submission.txt" 2>&1 \
      || { flock -u 8; fail "eval recovery submission failed"; }
  elif [[ "$class" != exists ]]; then
    flock -u 8; fail "eval recovery pre-existing state ambiguous"
  fi
  flock -u 8
  active_experiment="$next_experiment"; printf '%s\n' "$active_experiment" > "$active_file"
  log "monitoring eval recovery=$recovery_wave jobs=${failed_jobs[*]}"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
cp -a "$results_root/." "$state_dir/download/"
python scripts/summarize_layerwise_unified_final_eval.py \
  --root "$state_dir/download" --output "$state_dir/summary.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || { touch "$state_dir/scientific_gate_failed"; fail "final scientific gate failed"; }
touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
log "layerwise unified final evaluation complete"
