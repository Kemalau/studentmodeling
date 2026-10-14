#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/one_shot_layer_transport"
config="$project_dir/amlt/feeds_one_shot_layer_transport_c4_20260731.yaml"
base_experiment="recurrent-cf1-layer-transport-c4-seed42-20260731"
jobs=(cf1-visionpad-layer-transport-singles-n50-seed42 cf1-visionpad-layer-transport-blocks-n50-seed42 cf1-visionpad-layer-transport-span-n50-seed42)
mkdir -p "$state_dir/download"; cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"; trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

experiment="$base_experiment"; passed=false
for recovery in 0 1 2 3; do
  while true; do
    heartbeat; status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
    if python - "$status" "${#jobs[@]}" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
    then passed=true; break; fi
    break
  done
  [[ "$passed" == true ]] && break
  printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
  amlt results download "$experiment" --status pass --include layer_transport_c4_n50.json \
    --include scope.txt --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 || true
  timeout 180 amlt logs view -n 600 "$experiment" > "$state_dir/${experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]), sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 && recovery < 3 )) || break
  experiment="${base_experiment}-recovery$((recovery + 1))"; selectors=()
  for job in "${failed_jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$state_dir/${experiment}.dump.yaml" 2>&1 \
    || { flock -u 8; fail "layer-transport recovery dump failed"; }
  python scripts/validate_one_shot_layer_transport_amlt.py \
    --config "$state_dir/${experiment}.dump.yaml" --jobs "${failed_jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "layer-transport recovery preflight failed"; }
  output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Targeted infrastructure recovery for frozen layer-transport C4 screen" 2>&1)"; rc=$?
  printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"; flock -u 8
  (( rc == 0 )) || fail "layer-transport recovery submission failed"
  jobs=("${failed_jobs[@]}")
done
[[ "$passed" == true ]] || fail "layer-transport diagnostic exhausted recovery waves"

amlt results download "$experiment" --status pass --include layer_transport_c4_n50.json \
  --include scope.txt --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "layer-transport result download failed"
mapfile -t inputs < <(find "$state_dir/download" -type f -name layer_transport_c4_n50.json)
[[ ${#inputs[@]} -eq 3 ]] || fail "expected three layer-transport result files"
python scripts/summarize_one_shot_layer_transport.py --inputs "${inputs[@]}" \
  --output "$state_dir/summary.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "layer-transport summary validation failed"
touch "$state_dir/complete"; log "one-shot layer-transport C4 screen complete"
