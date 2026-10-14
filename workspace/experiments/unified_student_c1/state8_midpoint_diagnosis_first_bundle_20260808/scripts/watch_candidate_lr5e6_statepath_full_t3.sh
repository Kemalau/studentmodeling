#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
config="$project_dir/amlt/feeds_candidate_lr5e6_statepath_full_t3_20260804.yaml"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/t3_full_candidate_screen_20260804/lr5e6_statepath"
semantic_dir="$state_dir/t3_semantic"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
base_experiment=candidate-lr5e6-statepath-full-t3-seed42-20260804
job=candidate-lr5e6-statepath-full-t3-seed42
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"

mkdir -p "$state_dir" "$semantic_dir/full1494"
cd "$project_dir"
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
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if "$python_bin" - "$quota" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
q=select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']
assert int(q['standard']['limit'])-int(q['standard']['used']) >= 1
assert int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0) >= 1
PY
    then return 0; fi
    sleep 120
  done
}

submit_recovery(){
  local experiment="$1" current class output rc
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_candidate_lr5e6_statepath_full_t3_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$experiment" -t Feeds -y \
      --description "lr5e-6 full-T3 infrastructure recovery; hash-frozen candidate and protocol unchanged" 2>&1)"; rc=$?
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
}

if [[ ! -s "$state_dir/active_experiment.txt" ]]; then
  printf '%s\n' "$base_experiment" > "$state_dir/active_experiment.txt"
fi
experiment="$(<"$state_dir/active_experiment.txt")"
wave=0
if [[ "$experiment" =~ recovery([0-9]+)-20260804 ]]; then wave="${BASH_REMATCH[1]}"; fi

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
    >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|ServiceUnavailable' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 400 "$experiment" ":$job" \
    > "$state_dir/${experiment}.failure.log" 2>&1 || true
  (( wave < 5 )) || fail "lr5e-6 full-T3 exhausted five recovery waves"
  wave=$((wave+1))
  experiment="candidate-lr5e6-statepath-full-t3-recovery${wave}-20260804"
  submit_recovery "$experiment" || fail "lr5e-6 full-T3 recovery submission failed wave=$wave"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "lr5e-6 full-T3 result download failed"
readarray -t predictions < <(find "$state_dir/download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
readarray -t metrics < <(find "$state_dir/download" -type f -path '*/eval/t3/metrics.json' | sort)
[[ ${#predictions[@]} -eq 1 && ${#metrics[@]} -eq 1 ]] \
  || fail "expected one lr5e-6 T3 prediction/metrics pair"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "frozen T3 eval panel drift"
"$python_bin" - "${metrics[0]}" <<'PY' || fail "lr5e-6 full-T3 support audit failed"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,5120],x
assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False,x
assert x['sampled_rows']==1494,x
PY

exec 11>"$semantic_dir/judge.lock"; flock -x 11
(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
attempt=0
while [[ ! -f "$semantic_dir/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$semantic_dir/full1494" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$semantic_dir/cache.jsonl" --judge-audit-samples 100 --seed 42 \
      >> "$semantic_dir/full.log" 2>&1; then
    if "$python_bin" scripts/validate_t3_semantic_result.py \
        --metrics "$semantic_dir/full1494/metrics.json" \
        --output "$semantic_dir/t3_semantic_gate.json" \
        >> "$semantic_dir/controller.log" 2>&1; then
      touch "$semantic_dir/scientific_gate_pass"
    else
      rc=$?; (( rc == 2 )) || fail "lr5e-6 T3 semantic validator failed rc=$rc"
      touch "$semantic_dir/scientific_gate_failed"
    fi
    touch "$semantic_dir/complete"
    break
  fi
  (( attempt < 30 )) || fail "lr5e-6 T3 semantic judge exhausted"
  sleep 120
done
flock -u 11
touch "$state_dir/complete"
heartbeat
log "lr5e-6 state-path full-T3 screen completed"

