#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/known_oom_row_alignedmask"
experiment="layerwise-known-oom-row-alignedmask-recovery3-20260801"
job="layerwise-known-oom-row-recovery"

mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" ":$job" 2>&1)" || {
    sleep 60
    continue
  }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|statusCode[^0-9]*429|InternalServerError|ServiceUnavailable' <<< "$status"; then
    sleep 60
    continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60
    continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
  then
    break
  fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  timeout 90 amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
    > "$state_dir/failure.log" 2>&1 || true
  fail "aligned-mask known-OOM-row replay ended without a passed job"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include 'oom_recovery_receipt.json' --include 'completion.json' \
  --include 'trainer_history.json' --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "known-OOM-row result download failed"

python - "$state_dir/download" "$state_dir/receipt.json" <<'PY' \
  || fail "known-OOM-row receipt validation failed"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
receipts = list(root.rglob('oom_recovery_receipt.json'))
if len(receipts) != 1:
    raise ValueError(f'expected one OOM receipt, found {len(receipts)}')
value = json.loads(receipts[0].read_text(encoding='utf-8'))
assert value['status'] == 'passed', value
assert str(value['example_id']).endswith(':233:1'), value
assert value['direct_tokens'] == 549, value
assert value['direct_offload_used'] is True, value
assert float(value['peak_cuda_allocated_gib']) < 79.0, value
assert float(value['peak_cuda_reserved_gib']) < 79.0, value
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
print(json.dumps(value, indent=2, sort_keys=True))
PY
touch "$state_dir/complete"
log "aligned-mask known-OOM-row replay passed and receipt validated"
