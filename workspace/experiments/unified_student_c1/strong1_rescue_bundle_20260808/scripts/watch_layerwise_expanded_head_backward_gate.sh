#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/expanded_head_backward_gate"
final_fit="$pipeline/final_fit"
rehearsal="$pipeline/task_retention_rehearsal"
experiment="${EXPANDED_HEAD_GATE_EXPERIMENT:-layerwise-expanded-head-backward-finalizer-recovery2-seed42-20260802}"
runtime_sha="c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca"
mkdir -p "$state_dir"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)"; rc=$?
  printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$status" \
    >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|InternalServerError|ServiceUnavailable' <<< "$status"; then
    sleep 60; continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60; continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 3) else 1)
PY
  then break; fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  rm -rf "$state_dir/failure_logs"; mkdir -p "$state_dir/failure_logs"
  timeout 300 amlt logs download "$experiment" -F '*' -o "$state_dir/failure_logs" \
    >> "$state_dir/controller.log" 2>&1 || true
  fail "one or more exact-row expanded-head backward gates did not pass"
done

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include 'expanded_head_backward_receipt.json' --include 'trainer_history.json' \
  --include 'completion.json' --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "expanded-head gate result download failed"

python - "$state_dir/download" "$state_dir/receipt.json" "$runtime_sha" <<'PY' \
  || fail "expanded-head backward receipts failed validation"
import json, pathlib, sys
root, output, runtime = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
expected = {
    549: (177752, ':233:1'),
    653: (110499, ':0002079:malrule_identity'),
    721: (68865, ':0002130:malrule_identity'),
}
paths = list(root.rglob('expanded_head_backward_receipt.json'))
if len(paths) != 3:
    raise ValueError(paths)
rows = {}
for path in paths:
    value = json.loads(path.read_text(encoding='utf-8'))
    tokens = int(value['direct_tokens'])
    if tokens not in expected or tokens in rows:
        raise ValueError(value)
    index, suffix = expected[tokens]
    assert value['status'] == 'passed', value
    assert value['row_index'] == index, value
    assert str(value['example_id']).endswith(suffix), value
    assert value['direct_offload_used'] is True, value
    assert value['runtime_sha256'] == runtime, value
    assert float(value['peak_cuda_allocated_gib']) < 79.0, value
    assert float(value['peak_cuda_reserved_gib']) < 79.0, value
    rows[tokens] = value
payload = {
    'status': 'passed',
    'stage': 'expanded_head_sdpa_backward_gate',
    'runtime_sha256': runtime,
    'rows': [rows[token] for token in sorted(rows)],
}
temporary = output.with_suffix('.json.partial')
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
temporary.replace(output)
PY

test "$(sha256sum scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py | cut -d' ' -f1)" = "$runtime_sha" \
  || fail "validated runtime changed before controller release"
test "$(sha256sum tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py | cut -d' ' -f1)" = a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274 \
  || fail "validated runtime test changed before controller release"

rm -f "$final_fit/needs_attention" "$rehearsal/needs_attention" "$state_dir/needs_attention"
touch "$state_dir/complete"
heartbeat
log "all three real backward gates passed; releasing rehearsal recovery3 and main recovery10"

for session in layerwise_task_retention_rehearsal layerwise_unified_final_fit; do
  tmux kill-session -t "$session" 2>/dev/null || true
done
rm -f "$rehearsal/controller.active.pid" "$final_fit/controller.active.pid"
tmux new-session -d -s layerwise_task_retention_rehearsal \
  "cd '$project_dir' && exec bash scripts/watch_layerwise_task_retention_rehearsal.sh 2>&1 | tee -a '$rehearsal/controller.tmux.log'"
tmux new-session -d -s layerwise_unified_final_fit \
  "cd '$project_dir' && MAX_RESILIENT_RECOVERIES=12 exec bash scripts/watch_layerwise_unified_final_fit.sh 2>&1 | tee -a '$final_fit/controller.tmux.log'"
log "released both fail-closed controllers"
