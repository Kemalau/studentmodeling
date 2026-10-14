#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_midpoint_diagnosis_first_20260808"
config="$project/amlt/feeds_state8_midpoint_diagnosis_first_repair_20260808.yaml"
bundle="$project/state8_midpoint_diagnosis_first_bundle_20260808"
job=state8-midpoint-diagnosis-first-seed63
experiment_prefix=state8-midpoint-diagnosis-first-repair-recovery
experiment_suffix=-20260808
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
max_recovery_waves=24

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  [[ -z "$heartbeat_pid" ]] || { kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; }
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
heartbeat
(while true; do heartbeat; sleep 30; done) & heartbeat_pid=$!

phase(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
print(str(rows[0].get('status') or '').casefold() if len(rows)==1 else 'invalid')
PY
}

submit_recovery(){
  local experiment="$1"
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_midpoint_diagnosis_first_amlt.py \
    --config "$config" --bundle "$bundle" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" --dump -t Feeds > "$state/$experiment.preflight.yaml" \
    2> "$state/$experiment.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_midpoint_diagnosis_first_amlt.py \
    --config "$state/$experiment.preflight.yaml" --bundle "$bundle" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  amlt run "$config" "$experiment" -t Feeds -y \
    --description "Exact recovery of hash-bound State8 midpoint diagnosis-first repair" \
    > "$state/$experiment.submission.txt" 2>&1
  rc=$?; flock -u 8; return "$rc"
}

experiment="$(<"$state/active_experiment.txt")"; wave=0
[[ "$experiment" =~ recovery([0-9]+)-20260808 ]] && wave="${BASH_REMATCH[1]}"
while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  current="$(phase "$status")" || { sleep 120; continue; }
  case "$current" in
    pass) touch "$state/remote_passed" ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      sleep 120 ;;
    preparing|pre-queued|queued|running|resuming) sleep 120 ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_recovery_waves )) || fail "repair exhausted recovery waves"
      wave=$((wave+1)); experiment="${experiment_prefix}${wave}${experiment_suffix}"
      submit_recovery "$experiment" || fail "repair recovery submission failed wave=$wave"
      printf '%s\n' "$experiment" > "$state/active_experiment.txt"
      log "submitted repair recovery wave=$wave experiment=$experiment"
      ;;
  esac
done

if [[ ! -f "$state/results_downloaded" ]]; then
  rm -rf "$state/download"; mkdir -p "$state/download"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/download" >> "$state/controller.log" 2>&1 \
    || fail "repair result download failed"
  touch "$state/results_downloaded"
fi

mapfile -t formal < <(find "$state/download" -type f -path "*/$job/eval/formal_t3_300/predictions.jsonl")
mapfile -t diag_pred < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
mapfile -t diag_eval < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
mapfile -t diag_metrics < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/metrics.json")
[[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
  || fail "repair evaluation artifact cardinality drift"

if [[ ! -f "$state/t3_semantic_complete" ]]; then
  mkdir -p "$state/t3_semantic"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$state/t3_semantic" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/t3_semantic/cache.jsonl" --judge-audit-samples 100 \
    --seed 42 >> "$state/t3_semantic/judge.log" 2>&1 \
    || fail "repair T3 semantic judge failed"
  touch "$state/t3_semantic_complete"
fi

if [[ ! -f "$state/c4_semantic_complete" ]]; then
  mkdir -p "$state/c4_semantic"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
    --output-dir "$state/c4_semantic" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
    --seed 42 >> "$state/c4_semantic/judge.log" 2>&1 \
    || fail "repair diagnosis-C4 semantic judge failed"
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" \
    --output "$state/c4_semantic/summary.json" \
    >> "$state/c4_semantic/summarize.log" 2>&1 \
    || fail "repair diagnosis-C4 summary failed"
  touch "$state/c4_semantic_complete"
fi

set +e
"$python_bin" scripts/select_state8_midpoint_diagnosis_first.py \
  --root "$state" --output "$state/selection.json" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "repair selector crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "midpoint diagnosis-first repair reached terminal scientific selection"
