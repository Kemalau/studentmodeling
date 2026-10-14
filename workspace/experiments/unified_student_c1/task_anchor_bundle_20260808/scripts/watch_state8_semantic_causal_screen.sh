#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=state8-semantic-causal-strongbase-seed45-20260806
job=state8-semantic-causal-seed45
state="$project/results/state8_semantic_causal_screen_20260806"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep 120; continue; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then touch "$state/remote_passed"; break; fi
  printf '%s\n' "$status" > "$state/terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" > "$state/failure.log" 2>&1 || true
  fail "semantic causal job reached a non-pass terminal state"
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "semantic causal result download failed"
readarray -t formal < <(find "$state/download" -type f -path "*/$job/eval/formal_t3_300/predictions.jsonl")
readarray -t diag_pred < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
readarray -t diag_eval < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
readarray -t diag_metrics < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/metrics.json")
[[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
  || fail "semantic causal artifact cardinality drift"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
mkdir -p "$state/t3_semantic" "$state/c4_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state/t3_semantic" --judge-backend trapi \
  --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/t3_semantic/cache.jsonl" \
  --judge-audit-samples 100 --seed 45 >> "$state/t3_semantic/judge.log" 2>&1 \
  || fail "T3 semantic judge failed"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
  --output-dir "$state/c4_semantic" --judge-backend trapi \
  --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/c4_semantic/cache.jsonl" \
  --judge-audit-samples 100 --seed 42 >> "$state/c4_semantic/judge.log" 2>&1 \
  || fail "C4 semantic judge failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "${diag_metrics[0]}" \
  --output "$state/c4_semantic/summary.json" \
  >> "$state/c4_semantic/controller.log" 2>&1 \
  || fail "C4 semantic summary failed"
touch "$state/complete"; heartbeat
log "state8 semantic causal screen and semantic judges completed"
