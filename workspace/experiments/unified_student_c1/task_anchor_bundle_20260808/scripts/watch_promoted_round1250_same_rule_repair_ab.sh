#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1250_same_rule_repair_ab"
config="$project_dir/amlt/feeds_promoted_round1250_same_rule_repair_ab_20260804.yaml"
base_experiment=promoted-round1250-same-rule-repair-ab-20260804
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"
jobs=(promoted-round1250-repair-base-seed42 promoted-round1250-repair-same-rule-seed42)
mkdir -p "$state_dir/preserved" "$state_dir/passed" "$state_dir/t3_semantic"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention" "$state_dir/replication_recommended"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

declare -A experiments waves
map="$state_dir/experiments.tsv"
if [[ -s "$map" ]]; then
  while IFS=$'\t' read -r job experiment wave; do
    experiments["$job"]="$experiment"; waves["$job"]="$wave"
  done < "$map"
else
  for job in "${jobs[@]}"; do experiments["$job"]="$base_experiment"; waves["$job"]=0; done
fi
save_map(){
  local temporary="$map.partial.$$" job
  : > "$temporary"
  for job in "${jobs[@]}"; do printf '%s\t%s\t%s\n' "$job" "${experiments[$job]}" "${waves[$job]}" >> "$temporary"; done
  mv "$temporary" "$map"
}
save_map

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
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
  amlt results download "$experiment" --status pass \
    --include '*.json' --include 'predictions.jsonl' \
    --include 'predictions_redacted.jsonl' --output "$state_dir/preserved" \
    >> "$state_dir/controller.log" 2>&1
}

submit_recovery(){
  local job="$1" wave="$2" experiment output rc current class
  experiment="promoted-round1250-same-rule-repair-${job#promoted-round1250-repair-}-recovery${wave}-20260804"
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_promoted_round1250_same_rule_repair_ab_amlt.py \
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
      --description "Round-1250 controlled same-rule repair failed-sibling recovery wave=$wave job=$job; frozen A/B recipe unchanged" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" > "$state_dir/${experiment}.submission.txt"
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
    printf '[%s] experiment=%s job=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$job" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|ServiceUnavailable' <<< "$status"; then sleep 60; continue 2; fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
    then
      preserve_passed "$experiment" || { sleep 60; continue 2; }
      printf 'experiment=%s\npassed_at=%s\n' "$experiment" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$state_dir/passed/$job"
      log "preserved passed arm=$job experiment=$experiment"
      continue
    fi
    if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_active_job
raise SystemExit(0 if has_active_job(sys.argv[1]) else 1)
PY
    then continue; fi
    timeout 180 amlt logs view -n 2000 "$experiment" ":$job" > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
    waves["$job"]=$((waves["$job"]+1)); (( waves["$job"] <= 6 )) || fail "recovery waves exhausted arm=$job"
    preserve_passed "$experiment" || true
    submit_recovery "$job" "${waves[$job]}" || fail "recovery submit failed arm=$job"
  done
  [[ "$all_pass" == true ]] && break
  sleep 60
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved/." "$state_dir/download/"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 || fail "eval manifest drift"
for job in "${jobs[@]}"; do
  variant=base; [[ "$job" == promoted-round1250-repair-same-rule-seed42 ]] && variant=same_rule
  mapfile -t predictions < <(find "$state_dir/download" -type f -path "*${job}*/eval/formal_screen/predictions.jsonl" | sort)
  [[ ${#predictions[@]} -eq 1 ]] || fail "expected one T3 prediction file arm=$job found=${#predictions[@]}"
  output="$state_dir/t3_semantic/$variant"; mkdir -p "$output"; heartbeat
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$output/judged" --judge-backend trapi --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state_dir/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
    >> "$output/live.log" 2>&1 || fail "T3 semantic judge failed arm=$job"
  "$python_bin" scripts/validate_checkpoint_t3_semantic_result.py \
    --metrics "$output/judged/metrics.json" --output "$output/gate.json" --checkpoint-round 1500 \
    >> "$output/live.log" 2>&1; rc=$?
  (( rc == 0 || rc == 2 )) || fail "T3 semantic validation error arm=$job rc=$rc"
done

"$python_bin" scripts/summarize_promoted_round1250_same_rule_repair_ab.py \
  --root "$state_dir/download" --t3-root "$state_dir/t3_semantic" \
  --output "$state_dir/summary.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "A/B summary failed"
if "$python_bin" - "$state_dir/summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['replication_recommended'] else 1)
PY
then touch "$state_dir/replication_recommended"; fi
touch "$state_dir/complete"; heartbeat
log "round-1250 same-rule repair A/B complete; recipe remains unpromoted pending replication"
