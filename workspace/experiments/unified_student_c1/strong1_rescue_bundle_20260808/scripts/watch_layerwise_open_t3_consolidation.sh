#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/open_t3_consolidation"
config="$project_dir/amlt/feeds_layerwise_open_t3_consolidation_20260803.yaml"
validator=scripts/validate_layerwise_open_t3_consolidation_amlt.py
base_experiment=layerwise-open-t3-consolidation-seed42-20260803
jobs=(
  layerwise-open-t3-consolidation-statekd-lr5e7-seed42
  layerwise-open-t3-consolidation-statepath-lr5e7-seed42
)
mkdir -p "$state_dir/preserved" "$state_dir/passed_receipts" \
  "$state_dir/pruned_receipts" "$state_dir/t3_semantic"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention" "$state_dir/scientific_gate_pass"

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

preserve_passed(){
  local experiment="$1"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include 'predictions.jsonl' \
    --include 'predictions_redacted.jsonl' --output "$state_dir/preserved" \
    >> "$state_dir/controller.log" 2>&1
}

declare -A arm_experiment
arm_map="$state_dir/arm_experiments.tsv"
if [[ -s "$arm_map" ]]; then
  while IFS=$'\t' read -r job experiment; do
    [[ -n "$job" && -n "$experiment" ]] && arm_experiment["$job"]="$experiment"
  done < "$arm_map"
else
  for job in "${jobs[@]}"; do arm_experiment["$job"]="$base_experiment"; done
fi

save_arm_map(){
  local tmp="$arm_map.tmp.$$" job
  : > "$tmp"
  for job in "${jobs[@]}"; do
    printf '%s\t%s\n' "$job" "${arm_experiment[$job]}" >> "$tmp"
  done
  mv "$tmp" "$arm_map"
}
save_arm_map

# The state-path arm reached round 3500 before the platform terminated its
# post-training phase.  A separately downloaded, hash-audited checkpoint has
# already completed the frozen 150-pair full C4 evaluation locally.  If that
# evidence proves the specificity gate failed, record a terminal scientific
# prune instead of spending another 4xA100 allocation on downstream gates that
# cannot change the all-gates decision.
statepath_job=layerwise-open-t3-consolidation-statepath-lr5e7-seed42
statepath_c4="$state_dir/local_statepath_round3500_c4_full/summary.json"
statepath_summary="$state_dir/state_path.summary.json"
if [[ -s "$statepath_c4" ]]; then
  if python scripts/record_layerwise_c4_pruned_candidate.py \
    --evidence "$statepath_c4" --output "$statepath_summary" \
    --candidate-id equal_t3_t6_open_t3__state_path__lr5e-07 \
    --loss-profile state_path --learning-rate 5e-7 --checkpoint-round 3500 \
    >> "$state_dir/controller.log" 2>&1; then
    printf 'job=%s\nreason=frozen_full_c4_specificity_gate_failed\nevidence=%s\npruned_at=%s\n' \
      "$statepath_job" "$statepath_c4" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$state_dir/pruned_receipts/$statepath_job.complete"
    log "scientifically pruned arm=$statepath_job from audited full-C4 evidence"
  fi
fi

submit_recovery(){
  local job="$1" wave="$2" experiment output rc
  experiment="layerwise-open-t3-consolidation-${job#layerwise-open-t3-consolidation-}-recovery${wave}-20260803"
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python "$validator" --config "$state_dir/${experiment}.preflight.dump.yaml" \
    --jobs "$job" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
    --description "Phase-2 open-T3 consolidation recovery=$wave job=$job; scientific recipe unchanged" 2>&1)"; rc=$?
  printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
    > "$state_dir/${experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || return 1
  arm_experiment["$job"]="$experiment"
  save_arm_map
}

declare -A waves
for job in "${jobs[@]}"; do waves["$job"]=0; done
while true; do
  heartbeat
  all_passed=true
  transient=false
  for job in "${jobs[@]}"; do
    receipt="$state_dir/passed_receipts/$job.complete"
    [[ -f "$receipt" ]] && continue
    [[ -f "$state_dir/pruned_receipts/$job.complete" ]] && continue
    all_passed=false
    experiment="${arm_experiment[$job]}"
    status="$(timeout 90 amlt status "$experiment" ":$job" 2>&1)" || { transient=true; break; }
    printf '[%s] experiment=%s job=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$job" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|InternalServerError|ServiceUnavailable' <<< "$status"; then
      transient=true; break
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
      log "recorded passed arm=$job experiment=$experiment"
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
    timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
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
    [[ -f "$state_dir/passed_receipts/$job.complete" \
      || -f "$state_dir/pruned_receipts/$job.complete" ]] || all_passed=false
  done
  [[ "$all_passed" == true ]] && break
  sleep 120
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved/." "$state_dir/download/"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
backend=responses
extra="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  python_bin=/home/v-yuejiali/fkft/.venv/bin/python
  backend=trapi
  extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi

for profile in state_kd state_path; do
  if [[ "$profile" == state_path && -f "$state_dir/pruned_receipts/$statepath_job.complete" ]]; then
    test -s "$statepath_summary" || fail "pruned state-path summary missing"
    continue
  fi
  if [[ "$profile" == state_kd ]]; then marker=statekd; else marker=statepath; fi
  prediction="$(find "$state_dir/download" -type f -path "*${marker}*/eval/formal_screen/predictions.jsonl" -print -quit)"
  [[ -n "$prediction" ]] || fail "formal predictions missing profile=$profile"
  output="$state_dir/t3_semantic/$profile"; mkdir -p "$output"
  PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$prediction" \
    --eval-jsonl data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl \
    --output-dir "$output" --judge-backend "$backend" --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
    --judge-audit-samples 75 --seed 42 >> "$output/live.log" 2>&1 \
    || fail "T3 semantic judge failed profile=$profile"
  python scripts/summarize_t3_seen_unseen.py \
    --train-jsonl data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl \
    --eval-jsonl data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl \
    --predictions-jsonl "$prediction" \
    --judgments-jsonl "$output/predictions_redacted.jsonl" \
    --output "$output/seen_unseen.json" >> "$output/live.log" 2>&1 \
    || fail "T3 seen/unseen failed profile=$profile"
  python scripts/summarize_layerwise_single_task_retention.py \
    --root "$state_dir/download" --t3-metrics "$output/metrics.json" \
    --branch equal_t3_t6_open_t3 --loss-profile "$profile" \
    --learning-rate 5e-7 --checkpoint-round 3500 \
    --output "$state_dir/$profile.summary.json" >> "$state_dir/controller.log" 2>&1 \
    || fail "candidate summary failed profile=$profile"
done

python scripts/summarize_layerwise_open_t3_consolidation.py \
  --state-kd "$state_dir/state_kd.summary.json" \
  --state-path "$state_dir/state_path.summary.json" \
  --output "$state_dir/selection.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "consolidation selection failed"
touch "$state_dir/complete"
python - "$state_dir/selection.json" <<'PY' && touch "$state_dir/scientific_gate_pass"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
raise SystemExit(0 if x.get('all_gates_pass') is True else 1)
PY
heartbeat
log "phase-2 consolidation evaluation and selection complete"
