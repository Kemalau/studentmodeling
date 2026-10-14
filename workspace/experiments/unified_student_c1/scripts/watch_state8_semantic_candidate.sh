#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
: "${EXPERIMENT:?EXPERIMENT is required}"
: "${JOB:?JOB is required}"
: "${CANDIDATE_SLUG:?CANDIDATE_SLUG is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
validator="${VALIDATOR:-scripts/validate_state8_semantic_causal_screen_amlt.py}"
base_experiment="$EXPERIMENT"
state="${STATE_DIR:-$project/results/state8_semantic_candidates_20260806/$CANDIDATE_SLUG}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
max_waves="${MAX_RECOVERY_WAVES:-16}"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-keyflip-feeds}"

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

submit_recovery(){
  local wave="$1" next="state8-${CANDIDATE_SLUG}-recovery${wave}-20260806"
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$CONFIG_PATH" ":$JOB" --dump -t Feeds \
    > "$state/${next}.preflight.yaml" 2> "$state/${next}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" "$validator" \
    --config "$state/${next}.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$CONFIG_PATH" ":$JOB" "$next" -t Feeds -y \
    --description "Automatic exact-arm recovery for State8 semantic candidate=$CANDIDATE_SLUG wave=$wave" \
    > "$state/${next}.submission.txt" 2>&1 \
    || { flock -u 8; return 1; }
  flock -u 8
  experiment="$next"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted recovery experiment=$experiment"
}

wave=0
experiment="$base_experiment"
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260806 ]] && wave="${BASH_REMATCH[1]}"
else
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
fi

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
  printf '%s\n' "$status" > "$state/${experiment}.terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state/${experiment}.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$JOB" \
    > "$state/${experiment}.failure.log" 2>&1 || true
  (( wave < max_waves )) || fail "candidate exhausted recovery waves"
  wave=$((wave+1))
  submit_recovery "$wave" || fail "candidate recovery submission failed wave=$wave"
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "candidate result download failed"
readarray -t formal < <(find "$state/download" -type f -path "*/$JOB/eval/formal_t3_300/predictions.jsonl")
readarray -t diag_pred < <(find "$state/download" -type f -path "*/$JOB/eval/diagnosis_c4_50/judge_predictions.jsonl")
readarray -t diag_eval < <(find "$state/download" -type f -path "*/$JOB/eval/diagnosis_c4_50/judge_eval.jsonl")
readarray -t diag_metrics < <(find "$state/download" -type f -path "*/$JOB/eval/diagnosis_c4_50/metrics.json")
readarray -t publish < <(find "$state/download" -type f -path "*/$JOB/metrics/publish_receipt.json")
taxonomy_receipts=()
if [[ "$CANDIDATE_SLUG" == taxonomy ]]; then
  readarray -t taxonomy_receipts < <(find "$state/download" -type f -path "*/$JOB/metrics/taxonomy_causal_repair.json")
fi
[[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 \
   && ${#diag_metrics[@]} -eq 1 && ${#publish[@]} -eq 1 ]] \
  || fail "candidate artifact cardinality drift"
if [[ "$CANDIDATE_SLUG" == taxonomy && ${#taxonomy_receipts[@]} -ne 1 ]]; then
  fail "taxonomy receipt cardinality drift"
fi

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
cp "${publish[0]}" "$state/publish_receipt.json"
if [[ "$CANDIDATE_SLUG" == taxonomy ]]; then
  cp "${taxonomy_receipts[0]}" "$state/taxonomy_causal_repair.json"
fi
touch "$state/complete"; heartbeat
log "candidate training, download, and semantic judges completed"
