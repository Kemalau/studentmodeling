#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/cf4_single_hedge"
config="$project_dir/amlt/feeds_m4_cf4_single_hedge_20260731.yaml"
base_experiment="recurrent-m4-cf4-donor-directional-single-gb4-seed42-20260731"
job="m4-causal-cf4-donor-directional-w1-single-gb4-step0to1000-seed42"
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
    if python - "$quota" "$state_dir/quota_before_submission.json" <<'PY'
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
  python scripts/validate_m4_cf4_single_hedge_amlt_config.py \
    --config "$state_dir/${experiment}.dump.yaml" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "dump preflight failed"; }
  output="$(amlt run "$config" "$experiment" -t Feeds -y \
    --description 'Single-A100 queue hedge for the frozen CF4 donor-directional screen; same first 1000 examples, 250 optimizer updates, effective batch four' 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || fail "submission failed"
}

experiment="$base_experiment"
passed=false
for recovery in 0 1 2 3; do
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
[[ "$passed" == true ]] || fail "single-A100 CF4 hedge exhausted recovery waves"

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include recurrent_causal_gate.json \
  --include validation_metrics.json \
  --include causal_validation_metrics.json \
  --include branch_training_summary.json \
  --include run_config.json --include completion.json \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "result download failed"

python - "$state_dir/download" "$state_dir/comparison.json" <<'PY' \
  || fail "result validation failed"
import json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
gates = list(root.rglob('recurrent_causal_gate.json'))
valids = list(root.rglob('validation_metrics.json'))
causals = list(root.rglob('causal_validation_metrics.json'))
summaries = list(root.rglob('branch_training_summary.json'))
configs = list(root.rglob('run_config.json'))
completions = list(root.rglob('completion.json'))
assert all(len(x) == 1 for x in (gates, valids, causals, summaries, configs, completions)), {
  'gates': gates, 'valids': valids, 'causals': causals,
  'summaries': summaries, 'configs': configs, 'completions': completions,
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
assert completion['status'] == 'diagnostic_complete', completion
assert completion['micro_steps'] == 1000 and completion['optimizer_steps'] == 250, completion
assert completion['causal_pairs_seen'] == 1000 and completion['unique_causal_pairs_seen'] == 1000, completion
assert config['distributed'] is False and config['gradient_accumulation_steps'] == 4, config
assert config['causal_objective_mode'] == 'donor_directional', config
assert config['causal_repeat_fraction'] == 1.0, config
row = {
  **training,
  'branch': 'cf4_donor_directional_w1_single_gb4',
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
  'stage': 'M4_CF4_multiquery_diagnostic',
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
log "single-A100 CF4 hedge complete"
