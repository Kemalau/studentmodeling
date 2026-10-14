#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
base_experiment=user-candidate-lr2e6-parallel-formal-eval-seed42-20260804
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan/user_candidate_parallel_eval_20260804"
semantic_dir="$state_dir/t3_semantic"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
config="$project_dir/amlt/feeds_user_candidate_lr2e6_parallel_eval_20260804.yaml"
active_file="$state_dir/active_experiment.txt"
jobs_file="$state_dir/active_jobs.txt"
preserved="$state_dir/preserved_results"
all_jobs=(
  user-candidate-lr2e6-t2-seed42
  user-candidate-lr2e6-t3-seed42
  user-candidate-lr2e6-t5-seed42
)
t1_metrics="$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan/user_candidate_full_eval_20260803/t1/metrics.json"
t6_metrics="$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan/download/layerwise-eedi-full-open-t3-lr-scan-recovery1-seed42-20260802/layerwise-eedi-full-open-t3-lr2e6-seed42/eval/t6_full/metrics.json"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state_dir" "$semantic_dir" "$preserved"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
preserve_passed(){
  local experiment="$1" status
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || return 1
  grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status" || return 0
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' --output "$preserved" \
    >> "$state_dir/controller.log" 2>&1
}
wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if "$python_bin" - "$quota" "$required" <<'PY'
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
submit_recovery(){
  local experiment="$1"; shift
  local names=("$@") selectors=() output current class rc
  for item in "${names[@]}"; do selectors+=(":$item"); done
  wait_quota "${#names[@]}"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_user_candidate_parallel_eval_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "${names[@]}" \
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
      --description "Exact lr2e-6 formal-eval failed-sibling recovery jobs=${names[*]}; frozen artifact and protocol unchanged" 2>&1)"; rc=$?
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
  printf '%s\n' "$experiment" > "$active_file"
  printf '%s\n' "${names[@]}" > "$jobs_file"
}

rm -f "$state_dir/complete" "$state_dir/needs_attention"
test -s "$t1_metrics" || fail "T1 metrics are missing"
test -s "$t6_metrics" || fail "T6 metrics are missing"
if [[ ! -s "$active_file" ]]; then
  printf '%s\n' "$base_experiment" > "$active_file"
  printf '%s\n' "${all_jobs[@]}" > "$jobs_file"
fi
experiment="$(<"$active_file")"
expected_jobs="$(wc -l < "$jobs_file")"
wave=0
if [[ "$experiment" =~ recovery([0-9]+)-20260804 ]]; then wave="${BASH_REMATCH[1]}"; fi

while true; do
  heartbeat
  status="$(timeout 45 amlt status "$experiment" 2>&1)" || { sleep 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60
    continue
  fi
  if "$python_bin" - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$experiment" || fail "could not preserve passed candidate results"
    break
  fi
  preserve_passed "$experiment" || fail "could not preserve passed candidate siblings"
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal candidate evaluation has no failed jobs"
  for job in "${failed[@]}"; do
    amlt logs view -n 400 "$experiment" ":$job" \
      > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
  done
  (( wave < 5 )) || fail "candidate evaluation exhausted five recovery waves"
  wave=$((wave+1))
  experiment="user-candidate-lr2e6-parallel-formal-eval-recovery${wave}-20260804"
  submit_recovery "$experiment" "${failed[@]}" \
    || fail "candidate failed-sibling recovery submission failed wave=$wave"
  expected_jobs="${#failed[@]}"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
cp -a "$preserved/." "$state_dir/download/"

"$python_bin" - "$t1_metrics" "$t6_metrics" "$state_dir/download" "$state_dir/metrics.json" <<'PY'
import json
import pathlib
import sys

t1_path = pathlib.Path(sys.argv[1])
t6_path = pathlib.Path(sys.argv[2])
download = pathlib.Path(sys.argv[3])
output = pathlib.Path(sys.argv[4])

payloads = [json.loads(t1_path.read_text()), json.loads(t6_path.read_text())]
task_to_path = {}
for path in download.rglob("metrics.json"):
    payload = json.loads(path.read_text())
    tasks = {name.split("/", 1)[0] for name in payload.get("metrics_by_group", {})}
    if len(tasks) == 1:
        task = next(iter(tasks))
        if task in {"predict_response", "diagnose_misconception", "predict_prospective_error"}:
            assert task not in task_to_path, (task, task_to_path[task], path)
            task_to_path[task] = path
            payloads.append(payload)

assert set(task_to_path) == {
    "predict_response", "diagnose_misconception", "predict_prospective_error"
}, task_to_path

groups = {}
for payload in payloads:
    assert payload["state_architecture"] == "layerwise", payload
    assert payload["raw_history_visible_to_decoder"] is False, payload
    assert payload["persistent_history_kv_cache"] is False, payload
    for name, values in payload["metrics_by_group"].items():
        assert name not in groups, name
        groups[name] = values

expected = {
    "predict_correctness": 10913,
    "predict_response": 3693,
    "diagnose_misconception": 1494,
    "predict_prospective_error": 4600,
    "predict_error_persistence": 976,
}
observed = {
    task: sum(int(values["n"]) for name, values in groups.items() if name.startswith(task + "/"))
    for task in expected
}
assert observed == expected, (observed, expected)

merged = dict(payloads[0])
merged["metrics_by_group"] = groups
merged["sampled_rows"] = sum(expected.values())
merged["component_metrics"] = {
    "T1": str(t1_path),
    "T6": str(t6_path),
    **{task: str(path) for task, path in sorted(task_to_path.items())},
}
temporary = output.with_suffix(".json.partial")
temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY

"$python_bin" scripts/summarize_method_small_formal_metrics.py \
  "$state_dir/metrics.json" --output "$state_dir/headline.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "headline summarization failed"

"$python_bin" - "$state_dir/headline.json" <<'PY' || fail "headline support audit failed"
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
assert {key:int(value["n"]) for key,value in payload.items()} == {
    "T1":10913,"T2":3693,"T3":1494,"T5":4600,"T6":976
}, payload
PY

readarray -t t3_predictions < <(find "$state_dir/download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
[[ ${#t3_predictions[@]} -eq 1 ]] || fail "expected one full T3 prediction file"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "frozen T3 eval panel drift"
judge_python=/home/v-yuejiali/.venvs/studentmodeling/bin/python; backend=responses; extra="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  judge_python=/home/v-yuejiali/fkft/.venv/bin/python; backend=trapi
  extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi
exec 11>"$semantic_dir/judge.lock"
flock -x 11
attempt=0
while [[ ! -f "$semantic_dir/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$extra" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${t3_predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$semantic_dir/full1494" --judge-backend "$backend" \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$semantic_dir/cache.jsonl" --judge-audit-samples 100 --seed 42 \
      >> "$semantic_dir/full.log" 2>&1; then
    if "$python_bin" scripts/validate_t3_semantic_result.py \
        --metrics "$semantic_dir/full1494/metrics.json" \
        --output "$semantic_dir/t3_semantic_gate.json" \
        >> "$semantic_dir/controller.log" 2>&1; then
      touch "$semantic_dir/scientific_gate_pass"
    else
      touch "$semantic_dir/scientific_gate_failed"
    fi
    touch "$semantic_dir/complete"
    break
  fi
  (( attempt < 30 )) || fail "candidate T3 semantic judge exhausted"
  sleep 120
done
flock -u 11

touch "$state_dir/complete"
heartbeat
log "parallel T2/T3/T5 evaluation downloaded, merged with T1/T6, semantically judged, and audited"
