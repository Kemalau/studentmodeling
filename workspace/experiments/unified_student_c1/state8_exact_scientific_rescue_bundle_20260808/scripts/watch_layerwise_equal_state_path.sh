#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_slug="${STATE_SLUG:-equal_state_path}"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/$state_slug"
config="$project_dir/${CONFIG_PATH:-amlt/feeds_layerwise_equal_state_path_20260802.yaml}"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
base_experiment="${BASE_EXPERIMENT:-layerwise-equal-t3-t6-state-path-seed42-20260802}"
recovery_prefix="${RECOVERY_PREFIX:-layerwise-equal-t3-t6-state-path}"
job="${JOB_NAME:-layerwise-equal-t3-t6-state-path-seed42}"
loss_profile="${LOSS_PROFILE_EXPECTED:-state_path}"
summary_branch="${SUMMARY_BRANCH_EXPECTED:-equal_t3_t6}"
validator="${VALIDATOR:-scripts/validate_layerwise_equal_state_path_amlt.py}"
description="${EXPERIMENT_DESCRIPTION:-Matched equal-T3/T6 state-path diagnostic}"
experiment_date="${EXPERIMENT_DATE:-20260802}"
summary_learning_rate="${SUMMARY_LEARNING_RATE:-}"
mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" <<'PY'
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

submit_recovery(){
  local wave="$1" next output rc current class
  next="${recovery_prefix}-recovery${wave}-seed42-${experiment_date}"
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds \
    > "$state_dir/${next}.preflight.dump.yaml" \
    2> "$state_dir/${next}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python "$validator" \
    --config "$state_dir/${next}.preflight.dump.yaml" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$next" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$next" -t Feeds -y \
      --description "$description recovery=$wave; scientific recipe unchanged" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${next}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${next}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$next" > "$state_dir/active_experiment.txt"
}

experiment="$base_experiment"
[[ ! -s "$state_dir/active_experiment.txt" ]] || experiment="$(<"$state_dir/active_experiment.txt")"
printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
wave=0
[[ ! "$experiment" =~ recovery([0-9]+)-seed42 ]] || wave="${BASH_REMATCH[1]}"
while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)"; rc=$?
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
    >> "$state_dir/status.log"
  class="$(python - "$rc" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || fail "could not classify state-path experiment status"
  if [[ "$class" == missing ]]; then
    # A newly introduced arm has no pre-existing base experiment.  Treat that
    # as the initial, idempotently guarded wave-0 submission rather than a
    # transient service failure; otherwise the controller sleeps forever.
    [[ "$experiment" == "$base_experiment" && "$wave" -eq 0 ]] \
      || fail "active state-path experiment disappeared experiment=$experiment"
    submit_recovery 0 || fail "initial state-path submission failed wave=0"
    experiment="$(<"$state_dir/active_experiment.txt")"
    log "monitoring initial state-path submission wave=0 experiment=$experiment"
    continue
  fi
  if [[ "$class" == ambiguous ]]; then
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|InternalServerError|ServiceUnavailable' <<< "$status"; then
      sleep 120; continue
    fi
    fail "ambiguous state-path experiment status rc=$rc experiment=$experiment"
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 120; continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
  then break; fi

  timeout 90 amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
    > "$state_dir/${experiment}.failure.log" 2>&1 || true
  (( wave < 4 )) || fail "equal-state diagnostic exhausted four recovery waves"
  wave=$((wave+1))
  submit_recovery "$wave" || fail "equal-state recovery submission failed wave=$wave"
  experiment="$(<"$state_dir/active_experiment.txt")"
  log "monitoring state-path recovery wave=$wave experiment=$experiment"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.safetensors' --include '*.pt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "equal-state result download failed"
test "$(find "$state_dir/download" -type f -name task_retention_training_receipt.json | wc -l)" -eq 1 \
  || fail "equal-state training receipt missing"
test "$(find "$state_dir/download" -type f -path '*/eval/c4/recurrent_causal_gate.json' | wc -l)" -eq 1 \
  || fail "equal-state C4 result missing"
test "$(find "$state_dir/download" -type f -path '*/eval/formal_screen/metrics.json' | wc -l)" -eq 1 \
  || fail "equal-state formal result missing"
test "$(find "$state_dir/download" -type f -path '*/eval/t6_full/metrics.json' | wc -l)" -eq 1 \
  || fail "equal-state T6 result missing"
prediction="$(find "$state_dir/download" -type f -path '*/eval/formal_screen/predictions.jsonl' -print -quit)"
test -n "$prediction" || fail "equal-state formal predictions missing"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python; backend=responses; extra="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  python_bin=/home/v-yuejiali/fkft/.venv/bin/python; backend=trapi
  extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi
mkdir -p "$state_dir/t3_semantic"
PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state_dir/t3_semantic" --judge-backend "$backend" --judge-workers 12 \
  --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
  --judge-audit-samples 75 --seed 42 >> "$state_dir/t3_semantic.log" 2>&1 \
  || fail "equal-state T3 judge failed"
"$python_bin" scripts/validate_checkpoint_t3_semantic_result.py \
  --metrics "$state_dir/t3_semantic/metrics.json" --output "$state_dir/t3_semantic/gate.json" \
  --checkpoint-round 750 >> "$state_dir/controller.log" 2>&1 || true
summary_lr_args=()
[[ -z "$summary_learning_rate" ]] || summary_lr_args=(--learning-rate "$summary_learning_rate")
python scripts/summarize_layerwise_single_task_retention.py \
  --root "$state_dir/download" --t3-metrics "$state_dir/t3_semantic/metrics.json" \
  --branch "$summary_branch" --loss-profile "$loss_profile" \
  "${summary_lr_args[@]}" --output "$state_dir/summary.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "equal-state summary failed"
touch "$state_dir/complete"
python - "$state_dir/summary.json" <<'PY' && touch "$state_dir/scientific_gate_pass"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
raise SystemExit(0 if x.get('all_gates_pass') is True else 1)
PY
heartbeat
log "equal-T3/T6 $loss_profile diagnostic complete"
