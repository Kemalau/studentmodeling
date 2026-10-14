#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/layerwise_formal_screen"
experiment="${LAYERWISE_FORMAL_EXPERIMENT:-recurrent-layerwise-formal-screen-seed42-20260731-recovery1}"
mkdir -p "$state_dir/download"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/needs_attention"

heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(amlt status "$experiment" 2>&1)"
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
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
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 4) else 1)
PY
  then
    break
  fi
  printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
  amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 1200 "$experiment" > "$state_dir/failure.log" 2>&1 || true
  fail "layerwise formal screen jobs did not pass"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include metrics.json --include predictions.jsonl \
  --include predictions_redacted.jsonl --include manifest_audit.json \
  --include scope.txt --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "layerwise formal screen result download failed"

python - "$state_dir/download" "$state_dir/summary.json" <<'PY'
import json, pathlib, sys
root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
paths = sorted(root.rglob('metrics.json'))
assert len(paths) == 4, paths
rows = []
for path in paths:
    value = json.load(open(path, encoding='utf-8'))
    assert value['state_architecture'] == 'layerwise', value
    assert value['state_shape'] == [64, 5120], value
    assert value['raw_history_visible_to_decoder'] is False, value
    assert value['persistent_history_kv_cache'] is False, value
    assert 'state_curriculum_stage_a_cf1' in value['adapter'], value
    assert 'full_trainvalid_no_t4_strictt6' in value['decode_adapter'], value
    rows.append({
        'path': str(path),
        'sampled_rows': int(value['sampled_rows']),
        'metrics_by_group': value['metrics_by_group'],
    })
scopes = sorted(root.rglob('scope.txt'))
assert len(scopes) == 4, scopes
for scope in scopes:
    text = scope.read_text(encoding='utf-8')
    for required in (
        'raw_history_visible_to_decoder=false',
        'persistent_history_kv_cache=false',
        'optimizer_steps=0',
    ):
        assert required in text, (scope, required)
payload = {
    'stage': 'zero_training_layerwise_formal_screen',
    'experiment': 'recurrent-layerwise-formal-screen-seed42-20260731-recovery1',
    'jobs_passed': 4,
    'state_shape': [64, 5120],
    'optimizer_steps': 0,
    'raw_history_visible_to_decoder': False,
    'persistent_history_kv_cache': False,
    'rows': rows,
}
tasks = {
    str(name).split('/', 1)[0]
    for row in rows
    for name in row['metrics_by_group']
}
required_tasks = {
    'predict_correctness', 'predict_response', 'diagnose_misconception',
    'predict_prospective_error', 'predict_error_persistence',
}
assert tasks == required_tasks, (tasks, required_tasks)
for row in rows:
    assert row['sampled_rows'] > 0, row
    for group in row['metrics_by_group'].values():
        assert int(group.get('n', 0)) > 0, group
        for key, value in group.items():
            if isinstance(value, (int, float)):
                assert value == value and abs(float(value)) < float('inf'), (key, value)
        if 'format_valid_rate' in group:
            assert float(group['format_valid_rate']) >= 0.95, group
payload['scientific_gate_pass'] = True
payload['tasks'] = sorted(tasks)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY

touch "$state_dir/scientific_gate_pass" "$state_dir/complete"
log "layerwise formal performance screen complete"
