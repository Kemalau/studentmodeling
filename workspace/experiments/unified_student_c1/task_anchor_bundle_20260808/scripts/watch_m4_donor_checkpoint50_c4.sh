#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
checkpoint_round="${CHECKPOINT_ROUND:-50}"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/donor_checkpoint${checkpoint_round}_c4"
config="$project_dir/amlt/feeds_m4_donor_w1_round${checkpoint_round}_c4_20260731.yaml"
base_experiment="recurrent-m4-donor-w1-checkpoint${checkpoint_round}-c4-seed42-20260731"
job="m4-donor-w1-checkpoint${checkpoint_round}-c4-seed42"
run_dir="/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m4_causal_donor_directional_sum_w1_seed42_round500_20260731"
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

wait_quota() {
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
q = select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']
assert int(q['standard']['limit']) - int(q['standard']['used']) >= 1
assert int(q['user_max']['limit']) - int(q['user_max'].get('used') or 0) >= 1
PY
    then return 0; fi
    sleep 120
  done
}

experiment_class() {
  local name="$1" output rc
  output="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  python - "$rc" "$output" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
}

experiment="$base_experiment"
passed=false
for recovery in 0 1 2 3; do
  if (( recovery > 0 )); then
    experiment="${base_experiment}-recovery${recovery}"
    wait_quota
    exec 8>"$project_dir/results/feeds_submission_global.lock"
    flock 8
    amlt run "$config" --dump -t Feeds > "$state_dir/${experiment}.dump.yaml" 2>&1 \
      || { flock -u 8; fail "recovery dump failed"; }
    python scripts/validate_checkpoint_c4_probe_amlt_config.py \
      --config "$state_dir/${experiment}.dump.yaml" --round "$checkpoint_round" \
      --run-dir "$run_dir" --job-name "$job" >> "$state_dir/controller.log" 2>&1 \
      || { flock -u 8; fail "recovery preflight failed"; }
    class="$(experiment_class "$experiment")" \
      || { flock -u 8; fail "could not classify recovery experiment"; }
    if [[ "$class" == missing ]]; then
      output="$(amlt run "$config" "$experiment" -t Feeds -y \
        --description "Infrastructure recovery $recovery for frozen donor-directional round-${checkpoint_round} C4; no weight update" 2>&1)"
      rc=$?
      printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
      (( rc == 0 )) || { flock -u 8; fail "recovery submission failed"; }
    elif [[ "$class" != exists ]]; then
      flock -u 8
      fail "ambiguous recovery experiment state"
    fi
    flock -u 8
  fi

  while true; do
    heartbeat
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then
      sleep 120
      continue
    fi
    if grep -Fqi 'No jobs found in the current experiment' <<< "$status"; then
      sleep 30
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    then
      passed=true
      break
    fi
    amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
    timeout 180 amlt logs view -n 1000 "$experiment" \
      > "$state_dir/${experiment}.failure.log" 2>&1 || true
    break
  done
  [[ "$passed" == true ]] && break
done
[[ "$passed" == true ]] || fail "round-${checkpoint_round} C4 exhausted recovery waves"

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include recurrent_causal_gate.json --include probe_scope.txt \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "result download failed"
mapfile -t gates < <(find "$state_dir/download" -type f -name recurrent_causal_gate.json)
[[ ${#gates[@]} -eq 1 ]] || fail "expected exactly one recurrent C4 gate"
python - "${gates[0]}" "$state_dir/summary.json" "$checkpoint_round" <<'PY' \
  || fail "round-${checkpoint_round} C4 summary validation failed"
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
assert x['raw_history_visible_to_decoder'] is False, x
assert x['persistent_history_kv_cache'] is False, x
s = x['summary']; c = s['clean']
row = {
    'round': int(sys.argv[3]),
    'D_specific': s['D_specific'],
    'D_damage': s['D_damage'],
    'clean_n': s['clean_n'],
    'swap_flip_to_A': c['swap']['flip_to_A_rate'],
    'random_flip_to_A': c['random']['flip_to_A_rate'],
    'noop_flip_to_A': c['noop']['flip_to_A_rate'],
    'samemal_flip_to_A': c['samemal']['flip_to_A_rate'],
    'scientific_gate_pass': s['scientific_gate_pass'],
    'experiment': sys.argv[1],
}
open(sys.argv[2], 'w', encoding='utf-8').write(
    json.dumps(row, indent=2, sort_keys=True) + '\n'
)
print(json.dumps(row, indent=2, sort_keys=True))
PY
touch "$state_dir/complete"
log "donor-directional round-${checkpoint_round} frozen C4 complete"
