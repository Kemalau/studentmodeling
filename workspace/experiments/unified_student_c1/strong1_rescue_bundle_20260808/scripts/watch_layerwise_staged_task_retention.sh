#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_slug="${STATE_SLUG:-staged_task_retention}"
state_dir="$pipeline/$state_slug"
config="$project_dir/${CONFIG_PATH:-amlt/feeds_layerwise_staged_task_retention_20260802.yaml}"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
train_jsonl="$project_dir/data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl"
base_experiment="${BASE_EXPERIMENT:-layerwise-staged-task-retention-standard-seed42-20260802}"
job="${JOB_NAME:-layerwise-staged-t3-t6-standard-seed42}"
loss_profile="${LOSS_PROFILE_EXPECTED:-standard}"
validator="${VALIDATOR:-scripts/validate_layerwise_staged_task_retention_amlt.py}"
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
  local wave="$1" experiment output rc current class
  experiment="${base_experiment%-seed42-20260802}-recovery${wave}-seed42-20260802"
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python "$validator" \
    --config "$state_dir/${experiment}.preflight.dump.yaml" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$experiment" -t Feeds -y \
      --description "Staged train-only T3/T6 diagnostic recovery=$wave; scientific recipe unchanged" 2>&1)"; rc=$?
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
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
}

experiment="$base_experiment"
[[ ! -s "$state_dir/active_experiment.txt" ]] || experiment="$(<"$state_dir/active_experiment.txt")"
printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
wave=0
[[ ! "$experiment" =~ recovery([0-9]+)-seed42 ]] || wave="${BASH_REMATCH[1]}"
while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
    >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|InternalServerError|ServiceUnavailable' <<< "$status"; then
    sleep 120; continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 120; continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  timeout 90 amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
    > "$state_dir/${experiment}.failure.log" 2>&1 || true
  (( wave < 5 )) || fail "staged task-retention diagnostic exhausted five recoveries"
  wave=$((wave+1))
  submit_recovery "$wave" || fail "staged task-retention recovery submission failed wave=$wave"
  experiment="$(<"$state_dir/active_experiment.txt")"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.safetensors' --include '*.pt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "staged task-retention result download failed"
prediction="$(find "$state_dir/download" -type f -path '*/eval/formal_screen/predictions.jsonl' -print -quit)"
test -n "$prediction" || fail "staged task-retention formal predictions missing"
test "$(find "$state_dir/download" -type f -path '*/eval/c4/recurrent_causal_gate.json' | wc -l)" -eq 1 \
  || fail "staged task-retention C4 result missing"
test "$(find "$state_dir/download" -type f -path '*/eval/t6_full/metrics.json' | wc -l)" -eq 1 \
  || fail "staged task-retention T6 result missing"

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
  || fail "staged task-retention T3 judge failed"
python scripts/summarize_t3_seen_unseen.py \
  --train-jsonl "$train_jsonl" --eval-jsonl "$eval_jsonl" \
  --predictions-jsonl "$prediction" \
  --judgments-jsonl "$state_dir/t3_semantic/predictions_redacted.jsonl" \
  --output "$state_dir/t3_semantic/seen_unseen.json" \
  >> "$state_dir/t3_semantic.log" 2>&1 \
  || fail "staged task-retention T3 seen/unseen diagnostic failed"
"$python_bin" scripts/validate_checkpoint_t3_semantic_result.py \
  --metrics "$state_dir/t3_semantic/metrics.json" --output "$state_dir/t3_semantic/gate.json" \
  --checkpoint-round 750 >> "$state_dir/controller.log" 2>&1 || true
python scripts/summarize_layerwise_single_task_retention.py \
  --root "$state_dir/download" --t3-metrics "$state_dir/t3_semantic/metrics.json" \
  --branch staged_t3_t6 --loss-profile "$loss_profile" --output "$state_dir/summary.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "staged task-retention summary failed"
touch "$state_dir/complete"
python - "$state_dir/summary.json" <<'PY' && touch "$state_dir/scientific_gate_pass"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
raise SystemExit(0 if x.get('all_gates_pass') is True else 1)
PY
heartbeat
log "staged task-retention diagnostic and independent T3 judgment complete"
