#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment="${STATE8_EXPERIMENT:-state8-natural-dualpath-samerule-v5-20260805}"
job=state8-natural-curriculum-seed42
state_dir="$project_dir/results/${experiment//-/_}"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"

mkdir -p "$state_dir"
cd "$project_dir" || exit 1
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 120
    continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
  then
    touch "$state_dir/remote_passed"
    break
  fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 500 "$experiment" ":$job" > "$state_dir/failure.log" 2>&1 || true
  fail "remote state8 job did not pass"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "result download failed"

readarray -t predictions < <(
  find "$state_dir/download" -type f -path '*/eval/formal_screen/predictions.jsonl' | sort
)
readarray -t metrics < <(
  find "$state_dir/download" -type f -path '*/eval/formal_screen/metrics.json' | sort
)
[[ ${#predictions[@]} -eq 1 && ${#metrics[@]} -eq 1 ]] \
  || fail "expected one formal-screen prediction/metrics pair"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = \
  a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "frozen evaluation panel drift"

mkdir -p "$state_dir/t3_semantic"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
attempt=0
while [[ ! -f "$state_dir/t3_semantic/complete" ]]; do
  heartbeat
  attempt=$((attempt + 1))
  if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$state_dir/t3_semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
      --judge-audit-samples 100 --seed 42 \
      >> "$state_dir/t3_semantic/judge.log" 2>&1; then
    touch "$state_dir/t3_semantic/complete"
    break
  fi
  (( attempt < 30 )) || fail "semantic judge exhausted"
  sleep 120
done

python - "$state_dir/t3_semantic/metrics.json" <<'PY' || exit 1
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x["generation_rows"] == x["judged_rows"] and x["judge_errors"] == 0, x
assert x["judge_backend"] == "trapi", x
for group in x["metrics_by_group"].values():
    assert group.get("llm_judge_exact_shortcuts", 0) == 0, group
PY

touch "$state_dir/complete"
heartbeat
log "state8 remote training/evaluation and semantic judge completed"
