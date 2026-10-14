#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_slug="${STATE_SLUG:-m2_seeded_cf4_screen}"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/$state_slug"
config="$project_dir/${CONFIG_PATH:-amlt/feeds_m2_seeded_cf4_screen_20260731.yaml}"
base_experiment="${BASE_EXPERIMENT:-recurrent-m2-seeded-cf4-donor-directional-single-gb4-seed42-20260731}"
expected_branch="${EXPECTED_BRANCH:-m2_seeded_donor_directional_cf4_w1_single_gb4}"
expected_objective="${EXPECTED_OBJECTIVE:-donor_directional}"
validator="${VALIDATOR:-scripts/validate_m2_seeded_cf4_screen_amlt.py}"
max_recovery="${MAX_RECOVERY:-6}"
start_recovery="${START_RECOVERY:-0}"
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

experiment_class() {
  local name="$1" output rc
  output="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  python - "$rc" "$output" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
}

wait_quota() {
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$state_dir/quota_before_recovery.json" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1]))
q = row['quotas_(used/limit)']
available = int(q['standard']['limit']) - int(q['standard']['used'])
user_available = int(q['user_max']['limit']) - int(q['user_max'].get('used') or 0)
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps({
  'selected_row': row, 'required_units': 1,
  'standard_available': available, 'user_available': user_available,
}, indent=2, sort_keys=True) + '\n')
assert available >= 1 and user_available >= 1
PY
    then return 0; fi
    sleep 120
  done
}

submit_if_missing() {
  local experiment="$1" class output rc
  class="$(experiment_class "$experiment")" || fail "could not classify $experiment"
  [[ "$class" != exists ]] || return 0
  [[ "$class" == missing ]] || fail "ambiguous experiment state: $experiment"
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" --dump -t Feeds > "$state_dir/${experiment}.dump.yaml" 2>&1 \
    || { flock -u 8; fail "AMLT dump failed"; }
  python "$validator" \
    --config "$state_dir/${experiment}.dump.yaml" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "dump preflight failed"; }
  output="$(amlt run "$config" "$experiment" -t Feeds -y \
    --description 'Infrastructure recovery for M2 CF1-seeded CF4 single-A100 screen; strict state-only decoder and fixed seed42 protocol' 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || fail "submission failed"
}

experiment="$base_experiment"
passed=false
for ((recovery=start_recovery; recovery<=max_recovery; recovery++)); do
  (( recovery == 0 )) || experiment="${base_experiment}-recovery${recovery}"
  submit_if_missing "$experiment"
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
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    then passed=true; break; fi
    printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
    timeout 180 amlt logs view -n 600 "$experiment" \
      > "$state_dir/${experiment}.failure.log" 2>&1 || true
    break
  done
  [[ "$passed" == true ]] && break
done
[[ "$passed" == true ]] || fail "M2-seeded CF4 screen exhausted recovery waves"

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include recurrent_causal_gate.json \
  --include validation_metrics.json \
  --include causal_validation_metrics.json \
  --include branch_training_summary.json \
  --include run_config.json --include completion.json \
  --include init_provenance.txt \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "result download failed"

python - "$state_dir/download" "$state_dir/comparison.json" \
  "$expected_branch" "$expected_objective" <<'PY' \
  || fail "result validation failed"
import json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:3])
expected_branch = sys.argv[3]
expected_objective = sys.argv[4]
gates = list(root.rglob('recurrent_causal_gate.json'))
valids = list(root.rglob('validation_metrics.json'))
causals = list(root.rglob('causal_validation_metrics.json'))
summaries = list(root.rglob('branch_training_summary.json'))
configs = list(root.rglob('run_config.json'))
completions = list(root.rglob('completion.json'))
provenance = list(root.rglob('init_provenance.txt'))
assert all(len(x) == 1 for x in (gates, valids, causals, summaries, configs, completions, provenance)), {
  'gates': gates, 'valids': valids, 'causals': causals,
  'summaries': summaries, 'configs': configs, 'completions': completions,
  'provenance': provenance,
}
gate = json.loads(gates[0].read_text())
assert gate['raw_history_visible_to_decoder'] is False, gate
assert gate['persistent_history_kv_cache'] is False, gate
s = gate['summary']; c = s['clean']
validation = json.loads(valids[0].read_text())
causal = json.loads(causals[0].read_text())
training = json.loads(summaries[0].read_text())
config = json.loads(configs[0].read_text())
completion = json.loads(completions[0].read_text())
provenance_text = provenance[0].read_text()
assert 'initialization=m2_cf1_causal_adapter_and_state' in provenance_text, provenance_text
assert completion['status'] == 'diagnostic_complete', completion
assert completion['micro_steps'] == 1000 and completion['optimizer_steps'] == 250, completion
assert completion['causal_pairs_seen'] == 1000 and completion['unique_causal_pairs_seen'] == 1000, completion
assert config['distributed'] is False and config['gradient_accumulation_steps'] == 4, config
assert config['causal_objective_mode'] == expected_objective, config
assert config['causal_repeat_fraction'] == 1.0, config
assert training['branch'] == expected_branch, training
row = {
  **training,
  'branch': expected_branch,
  'initialization': 'm2_cf1_causal_adapter_and_state',
  'C4_D_specific': float(s['D_specific']),
  'C4_D_damage': float(s['D_damage']),
  'C4_clean_n': int(s['clean_n']),
  'C4_swap_flip': float(c['swap']['flip_to_A_rate']),
  'C4_random_flip': float(c['random']['flip_to_A_rate']),
  'C4_noop_flip': float(c['noop']['flip_to_A_rate']),
  'C4_same_rule_flip': float(c['samemal']['flip_to_A_rate']),
  'validation_accuracy_gap': float(validation['accuracy_gap_recurrent_minus_direct']),
  'validation_recurrent_accuracy': float(validation['recurrent_accuracy']),
  'causal_valid_directed_flip': float(causal['directed_flip_to_A_on_clean']),
  'topology': 'single_a100_gradient_accumulation_4',
  'screen_only': True,
}
eligible = (
  row['C4_clean_n'] >= 50 and row['C4_noop_flip'] <= .05
  and row['C4_same_rule_flip'] <= .10 and row['validation_accuracy_gap'] >= -.05
)
payload = {
  'stage': f'M2_seeded_CF4_{expected_objective}_diagnostic',
  'branches': [row],
  'selected_branch': row['branch'] if eligible and row['C4_D_specific'] >= .15 else None,
  'scientific_gate_pass': bool(eligible and row['C4_D_specific'] >= .15),
  'screen_only': True,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if python - "$state_dir/comparison.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
raise SystemExit(0 if x.get('scientific_gate_pass') is True else 1)
PY
then
  rm -f "$state_dir/scientific_gate_failed"
  touch "$state_dir/scientific_gate_pass"
else
  rm -f "$state_dir/scientific_gate_pass"
  touch "$state_dir/scientific_gate_failed"
fi
touch "$state_dir/complete"
log "M2-seeded CF4 screen complete"
