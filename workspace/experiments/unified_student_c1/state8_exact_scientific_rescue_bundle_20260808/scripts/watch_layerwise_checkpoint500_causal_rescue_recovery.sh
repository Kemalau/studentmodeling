#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/checkpoint500_causal_rescue_recovery"
config="$project_dir/amlt/feeds_layerwise_checkpoint500_causal_rescue_20260801.yaml"
validator="$project_dir/scripts/validate_layerwise_checkpoint500_causal_rescue_amlt.py"
base_experiment="layerwise-c500-causal-rescue-seed42-20260801"
strong_experiment="layerwise-c500-causal-rescue-recovery1-seed42-20260801"
dense_experiment="layerwise-c500-causal-rescue-recovery2-seed42-20260801"
strong_job="layerwise-c500-rescue-strong-same-density-seed42"
dense_job="layerwise-c500-rescue-dense-same-total-seed42"

mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

job_status(){
  local experiment="$1" job="$2"
  amlt status "$experiment" ":$job" 2>&1
}

job_passed(){
  python - "$1" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
}

job_active(){
  grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$1"
}

wait_job_terminal(){
  local experiment="$1" job="$2" label="$3" status
  while true; do
    heartbeat
    status="$(job_status "$experiment" "$job")"
    printf '[%s] label=%s experiment=%s\n%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" "$experiment" "$status" \
      >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|statusCode[^0-9]*429|InternalServerError|ServiceUnavailable' <<< "$status"; then
      sleep 120
      continue
    fi
    if job_active "$status"; then
      sleep 120
      continue
    fi
    printf '%s\n' "$status"
    return 0
  done
}

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$state_dir/quota_before_dense_recovery.json" <<'PY'
import json,sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(sys.argv[1])); q=row['quotas_(used/limit)']
available=int(q['standard']['limit'])-int(q['standard']['used'])
user_available=int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)
open(sys.argv[2],'w',encoding='utf-8').write(json.dumps({
    'selected_row':row,'required_units':4,
    'standard_available':available,'user_available':user_available,
},indent=2,sort_keys=True)+'\n')
assert available>=4 and user_available>=4,(available,user_available)
PY
    then return 0; fi
    sleep 120
  done
}

submit_dense_recovery(){
  local current rc class output
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  amlt run "$config" ":$dense_job" --dump -t Feeds \
    > "$state_dir/dense_recovery.preflight.dump.yaml" \
    2> "$state_dir/dense_recovery.preflight.stderr" \
    || { flock -u 8; fail "dense recovery dump failed"; }
  python "$validator" --config "$state_dir/dense_recovery.preflight.dump.yaml" \
    --jobs "$dense_job" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "dense recovery preflight failed"; }
  current="$(amlt status --no-update-db "$dense_experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; fail "dense recovery state is ambiguous"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" ":$dense_job" "$dense_experiment" -t Feeds -y \
      --description 'Infrastructure recovery2 for failed checkpoint-500 dense-same-total arm; define direct_token_count before offload/receipt use; scientific recipe unchanged' 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/dense_recovery.submission.txt"
    ((rc==0)) || { flock -u 8; fail "dense recovery submission failed"; }
  elif [[ "$class" != exists ]]; then
    flock -u 8
    fail "dense recovery pre-existing state is ambiguous"
  fi
  flock -u 8
}

# Reuse a passed original dense arm if it somehow completed.  Submit exactly
# one fixed-code recovery only after the original arm is unambiguously terminal.
# Check this arm first so its fixed-code recovery can queue while the already
# submitted strong recovery runs; the two independent conditions then overlap.
dense_status="$(wait_job_terminal "$base_experiment" "$dense_job" original_dense)"
if job_passed "$dense_status"; then
  selected_dense_experiment="$base_experiment"
  log "original dense arm passed"
else
  printf '%s\n' "$dense_status" > "$state_dir/original_dense.terminal_status.txt"
  amlt show "$base_experiment" > "$state_dir/original_dense.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$base_experiment" ":$dense_job" \
    > "$state_dir/original_dense.failure.log" 2>&1 || true
  wait_quota
  submit_dense_recovery
  dense_status="$(wait_job_terminal "$dense_experiment" "$dense_job" dense_recovery)"
  if ! job_passed "$dense_status"; then
    printf '%s\n' "$dense_status" > "$state_dir/dense_recovery.terminal_status.txt"
    amlt show "$dense_experiment" > "$state_dir/dense_recovery.show.txt" 2>&1 || true
    timeout 180 amlt logs view -n 3000 "$dense_experiment" ":$dense_job" \
      > "$state_dir/dense_recovery.failure.log" 2>&1 || true
    fail "dense recovery ended without a passed job"
  fi
  selected_dense_experiment="$dense_experiment"
  log "dense recovery passed"
fi

# The already-submitted strong recovery must pass; a second blind retry would
# hide a fresh failure mode, so fail closed if it does not.
strong_status="$(wait_job_terminal "$strong_experiment" "$strong_job" strong_recovery)"
job_passed "$strong_status" || {
  printf '%s\n' "$strong_status" > "$state_dir/strong_recovery.terminal_status.txt"
  amlt show "$strong_experiment" > "$state_dir/strong_recovery.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$strong_experiment" ":$strong_job" \
    > "$state_dir/strong_recovery.failure.log" 2>&1 || true
  fail "strong recovery ended without a passed job"
}
log "strong recovery passed"

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$strong_experiment" --status pass \
  --include 'recurrent_causal_gate.json' --include 'rescue_training_receipt.json' \
  --include 'completion.json' --include 'run_config.json' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "strong recovery result download failed"
amlt results download "$selected_dense_experiment" --status pass \
  --include 'recurrent_causal_gate.json' --include 'rescue_training_receipt.json' \
  --include 'completion.json' --include 'run_config.json' \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "dense result download failed"
python scripts/summarize_layerwise_checkpoint500_causal_rescue.py \
  --root "$state_dir/download" --output "$state_dir/comparison.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "recovered causal-rescue summary validation failed"
touch "$state_dir/complete"
log "checkpoint-500 causal-rescue recovery comparison complete"
