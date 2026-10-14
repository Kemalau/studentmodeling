#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
config=amlt/feeds_state8_high_causal_binding_20260806.yaml
validator=scripts/validate_state8_high_causal_binding_amlt.py
base_experiment=state8-highcausal-binding-screen-20260806
state="$project/results/state8_highcausal_binding_20260806"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds
  fi
fi

jobs=(state8-highcausal-joint-seed47 state8-highcausal-updater-seed48)
slugs=(joint updater)
mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

experiment="$base_experiment"
wave=0
[[ -s "$state/active_experiment.txt" ]] && experiment="$(<"$state/active_experiment.txt")"
if [[ "$experiment" =~ ^state8-highcausal-binding-recovery([0-9]+)-20260806$ ]]; then
  wave="${BASH_REMATCH[1]}"
elif [[ "$experiment" != "$base_experiment" ]]; then
  fail "unexpected active experiment=$experiment"
fi
printf '%s\n' "$experiment" > "$state/active_experiment.txt"

submit_recovery(){
  local requested_wave="${1:?recovery wave is required}"
  local next="state8-highcausal-binding-recovery${requested_wave}-20260806"
  exec 8>"$project/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds > "$state/${next}.preflight.yaml" \
    2> "$state/${next}.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" "$validator" --config "$state/${next}.preflight.yaml" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  amlt run "$config" "$next" -t Feeds -y \
    --description "Exact recovery of State8 high-causal matched screen wave=$requested_wave" \
    > "$state/${next}.submission.txt" 2>&1 || { flock -u 8; return 1; }
  flock -u 8
  experiment="$next"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted exact recovery experiment=$experiment"
}

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],2) else 1)
PY
  then break; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep 120; continue; fi
  printf '%s\n' "$status" > "$state/${experiment}.terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state/${experiment}.show.txt" 2>&1 || true
  for job in "${jobs[@]}"; do
    timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
      > "$state/${experiment}.${job}.failure.log" 2>&1 || true
  done
  (( wave < 16 )) || fail "high-causal screen exhausted recovery waves"
  wave=$((wave+1))
  submit_recovery "$wave" || fail "recovery submission failed wave=$wave"
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "result download failed"

for index in 0 1; do
  job="${jobs[$index]}" slug="${slugs[$index]}" arm="$state/$slug"
  mkdir -p "$arm/t3_semantic" "$arm/c4_semantic"
  mapfile -t formal < <(find "$state/download" -type f -path "*/$job/eval/formal_t3_300/predictions.jsonl")
  mapfile -t diag_pred < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
  mapfile -t diag_eval < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
  mapfile -t diag_metrics < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/metrics.json")
  mapfile -t publish < <(find "$state/download" -type f -path "*/$job/metrics/publish_receipt.json")
  [[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 \
     && ${#diag_metrics[@]} -eq 1 && ${#publish[@]} -eq 1 ]] \
    || fail "artifact cardinality drift arm=$slug"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$arm/t3_semantic" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$arm/t3_semantic/cache.jsonl" --judge-audit-samples 100 \
    --seed 47 >> "$arm/t3_semantic/judge.log" 2>&1 \
    || fail "T3 semantic judge failed arm=$slug"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
    --output-dir "$arm/c4_semantic" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$arm/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
    --seed 42 >> "$arm/c4_semantic/judge.log" 2>&1 \
    || fail "C4 semantic judge failed arm=$slug"
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$arm/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" \
    --output "$arm/c4_semantic/summary.json" \
    >> "$arm/c4_semantic/controller.log" 2>&1 \
    || fail "C4 semantic summary failed arm=$slug"
  cp "${publish[0]}" "$arm/publish_receipt.json"
done

taxonomy="$project/results/state8_highcausal_binding_20260806/taxonomy"
while [[ ! -f "$state/mid/complete" || ! -f "$state/eedi/complete" \
        || ! -f "$taxonomy/complete" ]]; do
  for candidate in "$state/mid" "$state/eedi" "$taxonomy"; do
    [[ ! -f "$candidate/needs_attention" ]] \
      || fail "candidate controller needs attention: $candidate"
  done
  heartbeat
  sleep 120
done
"$python_bin" scripts/select_state8_high_causal_candidate.py \
  --joint "$state/joint" --updater "$state/updater" --mid "$state/mid" \
  --eedi "$state/eedi" --taxonomy "$taxonomy" \
  --output "$state/selection.json" \
  >> "$state/controller.log" 2>&1 || fail "high-causal candidate selection failed"
az storage blob upload --auth-mode login \
  --account-name csnewsandfeeds4150361735 --container-name amulet \
  --name studentmodeling/gates/state8_highcausal_binding_v1/selection.json \
  --file "$state/selection.json" --overwrite \
  >> "$state/controller.log" 2>&1 || fail "promotion gate upload failed"
touch "$state/complete"; heartbeat
log "high-causal matched screen, semantic judging, and selection complete"
