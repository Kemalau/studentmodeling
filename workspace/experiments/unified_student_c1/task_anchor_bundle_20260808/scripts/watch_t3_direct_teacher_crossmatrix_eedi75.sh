#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
config="$project/amlt/feeds_t3_direct_teacher_crossmatrix_eedi75_20260804.yaml"
validator="$project/scripts/validate_t3_direct_teacher_crossmatrix_eedi75_amlt.py"
base_experiment=t3-direct-teacher-crossmatrix-eedi75-seed42-20260804
state_dir="$project/results/layerwise_unified_final_pipeline_20260731/t3_direct_teacher_crossmatrix_eedi75_20260804"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
all_jobs=(current-teacher-full-decoder-eedi75-seed42 fulltask-teacher-full-decoder-eedi75-seed42)

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
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if "$python_bin" - "$quota" "$required" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
q=select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']; n=int(sys.argv[2])
assert int(q['standard']['limit'])-int(q['standard']['used']) >= n
assert int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0) >= n
PY
    then return 0; fi
    sleep 120
  done
}

submit_recovery(){
  local experiment="$1"; shift
  local names=("$@") selectors=() current class output rc
  for name in "${names[@]}"; do selectors+=(":$name"); done
  wait_quota "${#names[@]}"
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$validator" --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "${names[@]}" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Direct-teacher Eedi-75 cross-matrix failed-sibling recovery; frozen artifacts/protocol; optimizer_steps=0" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" > "$state_dir/${experiment}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$state_dir/active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state_dir/active_jobs.txt"
}

if [[ ! -s "$state_dir/active_experiment.txt" ]]; then
  printf '%s\n' "$base_experiment" > "$state_dir/active_experiment.txt"
  printf '%s\n' "${all_jobs[@]}" > "$state_dir/active_jobs.txt"
fi
experiment="$(<"$state_dir/active_experiment.txt")"
wave=0
[[ ! "$experiment" =~ -recovery([0-9]+)-20260804 ]] || wave="${BASH_REMATCH[1]}"
while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s] experiment=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|ServiceUnavailable' <<< "$status"; then sleep 120; continue; fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
  expected="$(wc -l < "$state_dir/active_jobs.txt")"
  if "$python_bin" - "$status" "$expected" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],int(sys.argv[2])) else 1)
PY
  then break; fi
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal cross-matrix state has no failed jobs"
  for job in "${failed[@]}"; do
    timeout 180 amlt logs view -n 800 "$experiment" ":$job" > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
  done
  (( wave < 5 )) || fail "direct-teacher cross-matrix exhausted recovery waves"
  wave=$((wave+1)); experiment="${base_experiment%-20260804}-recovery${wave}-20260804"
  submit_recovery "$experiment" "${failed[@]}" || fail "cross-matrix recovery submission failed wave=$wave"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 || fail "cross-matrix result download failed"
readarray -t predictions < <(find "$state_dir/download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
[[ ${#predictions[@]} -eq 2 ]] || fail "expected two cross-matrix prediction files"
mkdir -p "$state_dir/t3_semantic"
(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
for prediction in "${predictions[@]}"; do
  job="$(basename "$(dirname "$(dirname "$(dirname "$prediction")")")")"
  out="$state_dir/t3_semantic/$job"
  mkdir -p "$out"
  attempt=0
  while [[ ! -f "$out/complete" ]]; do
    attempt=$((attempt+1)); heartbeat
    if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" --output-dir "$out" \
        --judge-backend trapi --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$state_dir/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
        >> "$out/judge.log" 2>&1; then
      touch "$out/complete"; break
    fi
    (( attempt < 30 )) || fail "semantic judge exhausted job=$job"
    sleep 120
  done
done
"$python_bin" - "$state_dir/summary.json" "${predictions[@]}" <<'PY' || fail "cross-matrix summary failed"
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); root=out.parent
rows=[]
for prediction in sys.argv[2:]:
    p=pathlib.Path(prediction); job=p.parents[2].name
    raw=json.load(open(p.parent/'metrics.json',encoding='utf-8'))
    judged=json.load(open(root/'t3_semantic'/job/'metrics.json',encoding='utf-8'))
    key='diagnose_misconception/eedi_mining/open_taxonomy_generation'
    rows.append({'job':job,'condition':raw['crossmatrix_condition'],'support':judged['judged_rows'],
                 'semantic_accuracy':judged['metrics_by_group'][key]['llm_judge_accuracy'],
                 'raw_history_visible_to_decoder':False,'optimizer_steps':0})
assert len(rows)==2 and {r['condition'] for r in rows}=={
    'current_teacher_full_decoder','fulltask_teacher_full_decoder'},rows
payload={'stage':'t3_direct_teacher_crossmatrix_eedi75_v1','diagnostic_only':True,'rows':sorted(rows,key=lambda x:x['condition'])}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
touch "$state_dir/complete"
heartbeat
log "direct-teacher Eedi-75 cross-matrix complete experiment=$experiment"
