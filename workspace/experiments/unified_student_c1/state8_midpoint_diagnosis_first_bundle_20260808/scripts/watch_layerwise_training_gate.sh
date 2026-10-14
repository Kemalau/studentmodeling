#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_training_gate"
experiment="${LAYERWISE_TRAINING_GATE_EXPERIMENT:-recurrent-layerwise-training-gate-seed42-20260731-recovery3}"
mkdir -p "$state_dir/download"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "$state_dir/controller.log"
}
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(amlt status "$experiment" 2>&1)"
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" \
    >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
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
  amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 2000 "$experiment" \
    > "$state_dir/failure.log" 2>&1 || true
  fail "layerwise training memory/gradient gate did not pass"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include completion.json --include run_config.json \
  --include gradient_audit.json --include trainer_history.json \
  --include validation_metrics.json --include causal_validation_metrics.json \
  --include data_audit.json --include scope.txt \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "layerwise gate result download failed"

python - "$state_dir/download" "$state_dir/summary.json" "$experiment" <<'PY' \
  || fail "layerwise gate result validation failed"
import json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:3])
experiment = sys.argv[3]

def one(name):
    paths = list(root.rglob(name))
    assert len(paths) == 1, (name, paths)
    return paths[0]

completion = json.loads(one('completion.json').read_text(encoding='utf-8'))
config = json.loads(one('run_config.json').read_text(encoding='utf-8'))
gradient = json.loads(one('gradient_audit.json').read_text(encoding='utf-8'))
history = json.loads(one('trainer_history.json').read_text(encoding='utf-8'))
validation = json.loads(one('validation_metrics.json').read_text(encoding='utf-8'))
causal = json.loads(one('causal_validation_metrics.json').read_text(encoding='utf-8'))
audit = json.loads(one('data_audit.json').read_text(encoding='utf-8'))
scope = one('scope.txt').read_text(encoding='utf-8')
assert completion['status'] == 'diagnostic_complete', completion
assert completion['micro_steps'] == 4 and completion['optimizer_steps'] == 4, completion
assert completion['contains_causal_pair'] is True, completion
assert config['state_architecture'] == 'layerwise', config
assert config['diagnostic_max_micro_steps'] == 4, config
assert config['gradient_accumulation_steps'] == 1, config
assert config['causal_objective_mode'] == 'donor_directional', config
assert config['distributed'] is False, config
assert gradient['lora']['gradient_abs_sum'] > 0, gradient
assert gradient['frozen_tensors_with_grad'] == [], gradient
assert len(history) == 4, history
assert any(row.get('causal_pair_id') for row in history), history
peak_allocated = max(float(row['peak_cuda_allocated_gib']) for row in history)
peak_reserved = max(float(row['peak_cuda_reserved_gib']) for row in history)
assert peak_reserved < 79.0, peak_reserved
assert audit['status'] == 'passed', audit
assert audit['train']['decoder_history_leaks'] == 0, audit
for required in (
    'raw_history_visible_to_decoder=false',
    'persistent_history_kv_cache=false',
    'state_architecture=one_address_layerwise_residual_trajectory',
):
    assert required in scope, (required, scope)
payload = {
    'stage': 'layerwise_training_memory_gradient_gate',
    'experiment': experiment,
    'scientific_gate_pass': True,
    'micro_steps': 4,
    'optimizer_steps': 4,
    'peak_cuda_allocated_gib': peak_allocated,
    'peak_cuda_reserved_gib': peak_reserved,
    'lora_gradient_abs_sum': float(gradient['lora']['gradient_abs_sum']),
    'validation': validation,
    'causal_validation': causal,
    'raw_history_visible_to_decoder': False,
    'persistent_history_kv_cache': False,
    'state_architecture': 'layerwise',
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY

touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
log "layerwise training memory/gradient gate complete"
