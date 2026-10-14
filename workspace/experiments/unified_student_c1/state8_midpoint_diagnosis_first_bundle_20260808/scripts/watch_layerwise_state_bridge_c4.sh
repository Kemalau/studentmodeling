#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_state_bridge"
experiment="recurrent-cf1-layerwise-state-bridge-c4-seed42-20260731"
mkdir -p "$state_dir/download"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(amlt status "$experiment" 2>&1)"
  printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
    sleep 120
    continue
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 120
    continue
  fi
  if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 2) else 1)
PY
  then
    break
  fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 1200 "$experiment" > "$state_dir/failure.log" 2>&1 || true
  fail "layerwise bridge jobs did not pass"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include layerwise_state_bridge.json --include scope.txt \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "layerwise bridge download failed"
python - "$state_dir/download" "$state_dir/summary.json" <<'PY'
import json, pathlib, sys
root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
paths = sorted(root.rglob('layerwise_state_bridge.json'))
assert len(paths) == 2, paths
rows = []
for path in paths:
    value = json.load(open(path, encoding='utf-8'))
    assert value['optimizer_steps'] == 0, value
    assert value['raw_history_visible_to_decoder'] is False, value
    assert value['persistent_history_kv_cache'] is False, value
    assert value['intervention_target'] == 'one_address_layerwise_residual_trajectory', value
    assert value['state_shape'][0] == 64, value
    summary = value['summary']
    rows.append({
        'position_mode': value['position_mode'],
        'D_specific': float(summary['D_specific']),
        'D_damage': float(summary['D_damage']),
        'clean_n': int(summary['clean_n']),
        'swap_flip_to_A': float(summary['clean']['swap']['flip_to_A_rate']),
        'random_flip_to_A': float(summary['clean']['random']['flip_to_A_rate']),
        'noop_flip_to_A': float(summary['clean']['noop']['flip_to_A_rate']),
        'same_rule_flip_to_A': float(summary['clean']['samemal']['flip_to_A_rate']),
        'scientific_gate_pass': bool(summary['scientific_gate_pass']),
        'source': str(path),
    })
assert {row['position_mode'] for row in rows} == {'compact', 'one_shot_aligned'}, rows
payload = {
    'stage': 'zero_training_layerwise_state_bridge',
    'scientific_gate_pass': any(row['scientific_gate_pass'] for row in rows),
    'rows': sorted(rows, key=lambda row: row['position_mode']),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(payload, indent=2, sort_keys=True))
PY
touch "$state_dir/complete"
log "layerwise state bridge complete"
