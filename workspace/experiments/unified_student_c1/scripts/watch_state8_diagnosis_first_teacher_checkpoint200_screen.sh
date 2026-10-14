#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_diagnosis_first_teacher_checkpoint200_screen_20260810"
experiment=state8-diagnosis-first-teacher-checkpoint200-screen-20260810
job=state8-diagnosis-first-teacher-checkpoint200-screen
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"; flock -n 9 || exit 0
echo $$ > "$state/watcher.active.pid"
trap 'rm -f "$state/watcher.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/watcher.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep_unlocked 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep_unlocked 60; continue; fi
  if grep -Eqi '\bpaused\b' <<< "$status"; then
    timeout 90 amlt resume "$experiment" ":$job" >> "$state/watcher.log" 2>&1 || true
    sleep_unlocked 60; continue
  fi
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" > "$state/failure.log" 2>&1 || true
  fail "checkpoint-200 diagnostic screen failed; final Teacher remains authoritative"
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' \
  --output "$state/download" >> "$state/watcher.log" 2>&1 || fail "checkpoint-200 result download failed"
mapfile -t predictions < <(find "$state/download" -type f -path '*/eval/t3/predictions.jsonl')
[[ ${#predictions[@]} -eq 1 ]] || fail "checkpoint-200 prediction cardinality drift"
mkdir -p "$state/semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state/semantic" --judge-backend trapi \
  --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
  --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/semantic/cache.jsonl" --judge-audit-samples 75 --seed 66 \
  >> "$state/semantic.log" 2>&1 || fail "checkpoint-200 TRAPI judge failed"
set +e
"$python_bin" scripts/validate_state8_diagnosis_first_teacher_screen.py \
  --metrics "$state/semantic/metrics.json" --output "$state/teacher_gate.json" \
  >> "$state/watcher.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "checkpoint-200 semantic validator crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "checkpoint-200 diagnostic semantic screen complete rc=$rc"
