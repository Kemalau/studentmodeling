#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment="${STATE8_DIAGNOSIS_C4_EXPERIMENT:-state8-v5-diagnosis-c4-smoke-recovery1-seed42-20260805}"
job=state8-v5-diagnosis-c4-smoke-seed42
state_dir="$project_dir/results/state8_v5_diagnosis_c4_smoke_20260805"
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state_dir"
cd "$project_dir" || exit 1
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
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
    sleep 120; continue
  fi
  if PYTHONPATH="$project_dir/scripts" "$python_bin" - "$status" <<'PY'
import sys
from recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then touch "$state_dir/remote_passed"; break; fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 1000 "$experiment" ":$job" > "$state_dir/failure.log" 2>&1 || true
  fail "remote diagnosis-C4 smoke did not pass"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "diagnosis-C4 result download failed"
readarray -t predictions < <(find "$state_dir/download" -type f -name judge_predictions.jsonl)
readarray -t evals < <(find "$state_dir/download" -type f -name judge_eval.jsonl)
readarray -t metrics < <(find "$state_dir/download" -type f -path '*/eval/diagnosis_c4/metrics.json')
[[ ${#predictions[@]} -eq 1 && ${#evals[@]} -eq 1 && ${#metrics[@]} -eq 1 ]] \
  || fail "diagnosis-C4 result cardinality drift"

mkdir -p "$state_dir/semantic"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
attempt=0
while [[ ! -f "$state_dir/semantic/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" --eval-jsonl "${evals[0]}" \
      --output-dir "$state_dir/semantic" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state_dir/semantic/cache.jsonl" \
      --judge-audit-samples 100 --seed 42 >> "$state_dir/semantic/judge.log" 2>&1; then
    touch "$state_dir/semantic/complete"; break
  fi
  (( attempt < 30 )) || fail "diagnosis-C4 semantic judge exhausted"
  sleep 120
done

PYTHONPATH="$project_dir/scripts" "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state_dir/semantic/predictions_redacted.jsonl" \
  --generation-metrics "${metrics[0]}" --output "$state_dir/diagnosis_c4_semantic.json" \
  >> "$state_dir/controller.log" 2>&1 || fail "diagnosis-C4 summary failed"
touch "$state_dir/complete"; heartbeat
log "state8 diagnosis-C4 smoke and semantic judge completed"
