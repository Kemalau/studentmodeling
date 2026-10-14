#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
pipeline_dir="$project_dir/results/recurrent_final_pipeline_20260730"
state_dir="$pipeline_dir/t3_semantic_judge"
raw_eval_jsonl="$project_dir/data/processed/formal_evaluation_v1/test.jsonl"
eval_dir="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2"
eval_jsonl="$eval_dir/test.jsonl"
expected_raw_eval_sha="bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8"
expected_eval_sha="a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525"
mkdir -p "$state_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
cd "$project_dir"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"
}

fail() {
  log "ERROR: $*"
  touch "$state_dir/needs_attention"
  exit 1
}

deadline=$(( $(date +%s) + 2592000 ))
while [[ ! -f "$pipeline_dir/final_eval_complete" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  [[ ! -f "$pipeline_dir/needs_attention" ]] || fail "Final recurrent pipeline needs attention"
  (( $(date +%s) < deadline )) || fail "Final evaluation wait exceeded 30 days"
  sleep 120
done

test -f "$raw_eval_jsonl" || fail "Frozen formal evaluation panel is missing"
test "$(sha256sum "$raw_eval_jsonl" | cut -d' ' -f1)" = "$expected_raw_eval_sha" \
  || fail "Frozen formal evaluation panel hash drift"
if [[ ! -f "$eval_jsonl" ]] \
  || [[ "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" != "$expected_eval_sha" ]]; then
  mkdir -p "$eval_dir"
  /home/v-yuejiali/.venvs/studentmodeling/bin/python \
    scripts/build_recurrent_evaluation_manifest.py \
    --input-jsonl "$raw_eval_jsonl" \
    --output-jsonl "$eval_jsonl" \
    --audit-json "$eval_dir/audit.json" \
    --max-updates 4 \
    --seed 20260730 >> "$state_dir/controller.log" 2>&1 \
    || fail "Could not build fixed-u4 T3 judge evidence manifest"
fi
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = "$expected_eval_sha" \
  || fail "Fixed-u4 T3 evidence manifest hash drift"

mapfile -t predictions < <(
  find "$pipeline_dir/final_eval_download" -type f -path '*/eval/t3/predictions.jsonl' | sort
)
[[ "${#predictions[@]}" -eq 1 ]] || fail "Expected one final T3 prediction file, found ${#predictions[@]}"
prediction="${predictions[0]}"

python_bin="/home/v-yuejiali/.venvs/studentmodeling/bin/python"
judge_backend="responses"
extra_pythonpath="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  python_bin="/home/v-yuejiali/fkft/.venv/bin/python"
  judge_backend="trapi"
  extra_pythonpath="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi
[[ -x "$python_bin" ]] || fail "Judge Python environment is missing: $python_bin"

common_cache="$state_dir/t3_judge_cache.jsonl"
smoke_dir="$state_dir/smoke16"
full_dir="$state_dir/full1494"
mkdir -p "$smoke_dir" "$full_dir"

smoke_attempt=0
while [[ ! -f "$state_dir/smoke_complete" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  smoke_attempt=$((smoke_attempt + 1))
  log "T3 semantic judge smoke attempt=$smoke_attempt backend=$judge_backend"
  if PYTHONPATH="$extra_pythonpath" "$python_bin" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$prediction" \
      --eval-jsonl "$eval_jsonl" \
      --output-dir "$smoke_dir" \
      --judge-backend "$judge_backend" \
      --judge-workers 4 \
      --judge-max-retries 1 \
      --judge-timeout-seconds 60 \
      --judge-cache "$common_cache" \
      --max-records 16 \
      --judge-audit-samples 16 \
      --seed 42 >> "$state_dir/smoke.log" 2>&1; then
    if python - "$smoke_dir/metrics.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["generation_rows"] == 16, value
assert value["judged_rows"] == 16, value
assert value["judge_errors"] == 0, value
groups = value["metrics_by_group"]
assert sum(int(group.get("llm_judge_api_or_cache", 0)) for group in groups.values()) >= 1, value
PY
    then
      touch "$state_dir/smoke_complete"
      log "T3 semantic judge smoke passed"
      break
    fi
  fi
  if (( smoke_attempt >= 720 )); then
    fail "T3 semantic judge smoke did not recover within 24 hours"
  fi
  sleep 120
done

full_attempt=0
while [[ ! -f "$state_dir/full_complete" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  full_attempt=$((full_attempt + 1))
  log "T3 semantic full-panel attempt=$full_attempt backend=$judge_backend"
  if PYTHONPATH="$extra_pythonpath" "$python_bin" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$prediction" \
      --eval-jsonl "$eval_jsonl" \
      --output-dir "$full_dir" \
      --judge-backend "$judge_backend" \
      --judge-workers 12 \
      --judge-max-retries 12 \
      --judge-timeout-seconds 60 \
      --judge-cache "$common_cache" \
      --judge-audit-samples 100 \
      --seed 42 >> "$state_dir/full.log" 2>&1; then
    if "$python_bin" scripts/validate_t3_semantic_result.py \
        --metrics "$full_dir/metrics.json" \
        --output "$state_dir/t3_semantic_gate.json" \
        >> "$state_dir/controller.log" 2>&1; then
      touch "$state_dir/full_complete"
      rm -f "$pipeline_dir/t3_semantic_judge_pending"
      touch "$pipeline_dir/t3_semantic_judge_complete"
      "$python_bin" scripts/validate_recurrent_final_pipeline.py \
        --root "$pipeline_dir" \
        --output "$pipeline_dir/overall_completion.json" \
        >> "$state_dir/controller.log" 2>&1 \
        || fail "Final pipeline aggregate audit failed"
      touch "$pipeline_dir/overall_complete"
      log "Final fixed-panel T3 semantic judge and aggregate pipeline audit passed"
      break
    fi
  fi
  if (( full_attempt >= 30 )); then
    fail "T3 semantic full-panel judge failed 30 resumable attempts"
  fi
  sleep 120
done
