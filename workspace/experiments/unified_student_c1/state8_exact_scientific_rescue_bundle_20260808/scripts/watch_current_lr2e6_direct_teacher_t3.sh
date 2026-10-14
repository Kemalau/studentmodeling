#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
config="$project/amlt/feeds_current_lr2e6_direct_teacher_t3_20260804.yaml"
validator="$project/scripts/validate_current_lr2e6_direct_teacher_t3_amlt.py"
base_experiment=current-lr2e6-direct-teacher-t3-eedi75-seed42-20260804
job=current-lr2e6-direct-teacher-t3-eedi75-seed42
state_dir="$project/results/layerwise_unified_final_pipeline_20260731/current_lr2e6_direct_teacher_t3_20260804"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"

mkdir -p "$state_dir"
cd "$project"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
heartbeat_pid=""
cleanup(){
  [[ -z "$heartbeat_pid" ]] || { kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; }
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
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds > "$state_dir/${experiment}.preflight.dump.yaml" 2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$validator" --config "$state_dir/${experiment}.preflight.dump.yaml" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "$experiment" -t Feeds -y --description "Direct-teacher Eedi T3 mechanism diagnostic recovery; frozen artifact/protocol; not final-gate eligible" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" > "$state_dir/${experiment}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
}

[[ -s "$state_dir/active_experiment.txt" ]] || printf '%s\n' "$base_experiment" > "$state_dir/active_experiment.txt"
experiment="$(<"$state_dir/active_experiment.txt")"
wave=0
[[ ! "$experiment" =~ -recovery([0-9]+)-20260804 ]] || wave="${BASH_REMATCH[1]}"
while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|ServiceUnavailable' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  timeout 180 amlt logs view -n 600 "$experiment" ":$job" > "$state_dir/${experiment}.failure.log" 2>&1 || true
  (( wave < 5 )) || fail "direct-teacher T3 exhausted five recovery waves"
  wave=$((wave+1)); experiment="${base_experiment%-20260804}-recovery${wave}-20260804"
  submit_recovery "$experiment" || fail "direct-teacher T3 recovery submission failed wave=$wave"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' --include '*.txt' --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 || fail "direct-teacher result download failed"
readarray -t predictions < <(find "$state_dir/download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
readarray -t metrics < <(find "$state_dir/download" -type f -path '*/eval/t3/metrics.json' | sort)
[[ ${#predictions[@]} -eq 1 && ${#metrics[@]} -eq 1 ]] || fail "expected one direct-teacher prediction/metrics pair"
"$python_bin" - "${metrics[0]}" <<'PY' || fail "direct-teacher support audit failed"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['sampled_rows']==75 and x['state_source']=='direct_teacher',x
assert x['diagnostic_only'] is True and x['eligible_for_final_gate'] is False,x
assert x['raw_history_visible_to_state_encoder'] is True,x
assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False,x
PY

semantic="$state_dir/t3_semantic"
mkdir -p "$semantic/eedi75"
exec 11>"$semantic/judge.lock"; flock -x 11
(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
if [[ ! -f "$semantic/complete" ]]; then
  attempt=0
  while true; do
    attempt=$((attempt+1)); heartbeat
    if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
        --output-dir "$semantic/eedi75" --judge-backend trapi \
        --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
        >> "$semantic/full.log" 2>&1; then break; fi
    (( attempt < 30 )) || fail "direct-teacher T3 judge exhausted"
    sleep 120
  done
  "$python_bin" - "$semantic/eedi75/metrics.json" "$semantic/summary.json" <<'PY' || fail "direct-teacher semantic audit failed"
import json,pathlib,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); groups=x['metrics_by_group']
name='diagnose_misconception/eedi_mining/open_taxonomy_generation'
assert x['generation_rows']==75 and x['judged_rows']==75 and x['judge_errors']==0,x
assert set(groups)=={name} and groups[name]['n']==75 and groups[name]['llm_judge_n']==75,groups
out={'stage':'direct_teacher_state_T3_mechanism_diagnostic_v1','diagnostic_only':True,
     'eligible_for_final_gate':False,'support':75,
     'eedi_semantic_accuracy':groups[name]['llm_judge_accuracy'],
     'judge_backend':x['judge_backend'],'judge_rubric_sha256':x['judge_rubric_sha256']}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
PY
  touch "$semantic/complete"
fi
flock -u 11
touch "$state_dir/complete"
heartbeat
log "direct-teacher T3 mechanism diagnostic complete experiment=$experiment"
