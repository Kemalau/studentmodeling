#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/m2_m3_c4_transition"
config="$project_dir/amlt/feeds_m2_m3_c4_transition_20260731.yaml"
base_experiment="recurrent-m2-m3-c4-transition-seed42-20260731"
jobs=(m2-cf1-native-c4-seed42 m2-state-full-adapter-c4-seed42 m3-generation-kd-c4-seed42)
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

download_passed() {
  local experiment="$1"
  amlt results download "$experiment" --status pass \
    --include recurrent_causal_gate.json --include transition_provenance.txt \
    --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1
}

experiment="$base_experiment"
passed=false
for recovery in 0 1 2 3; do
  while true; do
    heartbeat
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
      sleep 120; continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120; continue
    fi
    if python - "$status" "${#jobs[@]}" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
    then
      download_passed "$experiment" || fail "transition result download failed"
      passed=true
      break
    fi
    break
  done
  [[ "$passed" == true ]] && break
  printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
  download_passed "$experiment" || true
  timeout 180 amlt logs view -n 600 "$experiment" \
    > "$state_dir/${experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]), sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal transition state has no recoverable jobs"
  (( recovery < 3 )) || break
  experiment="${base_experiment}-recovery$((recovery + 1))"
  selectors=(); for job in "${failed_jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.dump.yaml" 2>&1 \
    || { flock -u 8; fail "transition recovery dump failed"; }
  python scripts/validate_m2_m3_c4_transition_amlt.py \
    --config "$state_dir/${experiment}.dump.yaml" --jobs "${failed_jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "transition recovery preflight failed"; }
  output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Targeted infrastructure recovery for M2-to-M3 frozen C4 transition audit" 2>&1)"
  rc=$?; printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || fail "transition recovery submission failed"
  jobs=("${failed_jobs[@]}")
done
[[ "$passed" == true ]] || fail "M2-to-M3 transition audit exhausted recovery waves"

python - "$state_dir/download" "$state_dir/summary.json" <<'PY' \
  || fail "transition result validation failed"
import json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
gates = list(root.rglob('recurrent_causal_gate.json'))
provenance = list(root.rglob('transition_provenance.txt'))
assert len(gates) == 3 and len(provenance) == 3, (gates, provenance)
rows = []
for gate_path in gates:
    job = next(part for part in gate_path.parts if part.endswith('-c4-seed42'))
    gate = json.loads(gate_path.read_text())
    assert gate['raw_history_visible_to_decoder'] is False, gate
    assert gate['persistent_history_kv_cache'] is False, gate
    s = gate['summary']; c = s['clean']
    rows.append({
        'job': job,
        'D_specific': float(s['D_specific']),
        'D_damage': float(s['D_damage']),
        'clean_n': int(s['clean_n']),
        'swap': float(c['swap']['flip_to_A_rate']),
        'random': float(c['random']['flip_to_A_rate']),
        'noop': float(c['noop']['flip_to_A_rate']),
        'same_rule': float(c['samemal']['flip_to_A_rate']),
        'scientific_gate_pass': bool(s['scientific_gate_pass']),
    })
by_job = {row['job']: row for row in rows}
assert set(by_job) == {
    'm2-cf1-native-c4-seed42',
    'm2-state-full-adapter-c4-seed42',
    'm3-generation-kd-c4-seed42',
}, by_job
m2 = by_job['m2-cf1-native-c4-seed42']['D_specific']
reset = by_job['m2-state-full-adapter-c4-seed42']['D_specific']
m3 = by_job['m3-generation-kd-c4-seed42']['D_specific']
if m2 >= .15 and reset < .15:
    diagnosis = 'adapter_reset_erases_causal_decoder'
elif m2 >= .15 and reset >= .15 and m3 < .15:
    diagnosis = 'm3_training_erases_causal_state'
elif m2 < .15:
    diagnosis = 'recurrent_m2_never_closed_causal_state'
else:
    diagnosis = 'm3_preserves_causal_state'
payload = {
    'stage': 'M2_M3_frozen_C4_transition_audit',
    'rows': rows,
    'diagnosis': diagnosis,
    'm2_to_reset_delta': reset - m2,
    'reset_to_m3_delta': m3 - reset,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY
touch "$state_dir/complete"
log "M2-to-M3 frozen C4 transition audit complete"
