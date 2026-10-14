#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/recompute_mask_release"
final_fit="$pipeline/final_fit"
experiment="layerwise-known-oom-row-recompute-mask-recovery5-20260802"
job="layerwise-known-oom-row-recovery"
mkdir -p "$state_dir"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  heartbeat
  if ! status="$(timeout 90 amlt status "$experiment" 2>&1)"; then
    sleep 60
    continue
  fi
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|InternalServerError|ServiceUnavailable' <<< "$status"; then
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
  timeout 180 amlt logs view -n 3000 "$experiment" ":$job" > "$state_dir/failure.log" 2>&1 || true
  fail "real Qwen3 549-token checkpoint-recompute backward did not pass; main 16-GPU recovery remains held"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include 'oom_recovery_receipt.json' --include 'completion.json' \
  --include 'trainer_history.json' --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "real backward receipt download failed"

python - "$state_dir/download" "$state_dir/receipt.json" <<'PY' \
  || fail "real backward receipt validation failed"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
receipts = list(root.rglob('oom_recovery_receipt.json'))
histories = list(root.rglob('trainer_history.json'))
if len(receipts) != 1 or len(histories) != 1:
    raise ValueError((receipts, histories))
receipt = json.loads(receipts[0].read_text(encoding='utf-8'))
history = json.loads(histories[0].read_text(encoding='utf-8'))
assert receipt['status'] == 'passed', receipt
assert str(receipt['example_id']).endswith(':233:1'), receipt
assert receipt['direct_tokens'] == 549, receipt
assert receipt['direct_offload_used'] is True, receipt
assert float(receipt['peak_cuda_allocated_gib']) < 79.0, receipt
assert float(receipt['peak_cuda_reserved_gib']) < 79.0, receipt
assert len(history) == 1, history
assert history[0]['example_id'] == receipt['example_id'], (history, receipt)
assert history[0]['direct_tokens'] == 549, history
assert history[0]['direct_offload_used'] is True, history
path = pathlib.Path(sys.argv[2])
temporary = path.with_suffix('.json.partial')
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
temporary.replace(path)
PY

test "$(sha256sum scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py | cut -d' ' -f1)" \
  = c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca \
  || fail "validated runtime source changed before release"
test "$(sha256sum tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py | cut -d' ' -f1)" \
  = a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274 \
  || fail "validated runtime test changed before release"

rm -f "$final_fit/needs_attention"
touch "$state_dir/complete"
log "real backward receipt passed; released main exact-resume controller"

if tmux has-session -t layerwise_unified_final_fit 2>/dev/null; then
  tmux kill-session -t layerwise_unified_final_fit 2>/dev/null || true
fi
rm -f "$final_fit/controller.active.pid"
tmux new-session -d -s layerwise_unified_final_fit \
  "cd '$project_dir' && MAX_RESILIENT_RECOVERIES=12 exec bash scripts/watch_layerwise_unified_final_fit.sh 2>&1 | tee -a '$final_fit/controller.tmux.log'"
log "started main controller; it will monitor the already-submitted validated recovery9"
