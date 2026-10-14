#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/directional_checkpoint250_c4"
base_config="$project_dir/amlt/feeds_m4_checkpoint_c4_probe_20260731.yaml"
account="csnewsandfeeds4150361735"
container="amulet"
expected_tenant="72f988bf-86f1-41af-91ab-2d7cd011db47"
expected_subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
branches=(bootstrap_balanced_sum_w1 bootstrap_balanced_sum_w4)
mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
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

wait_auth() {
  while true; do
    heartbeat
    az account set --subscription "$expected_subscription" >/dev/null 2>&1 \
      || { sleep 60; continue; }
    value="$(az account show --query '{tenant:tenantId,subscription:id,state:state}' -o json 2>/dev/null || true)"
    if python - "$value" "$expected_tenant" "$expected_subscription" <<'PY'
import json, sys
x = json.loads(sys.argv[1])
assert x['tenant'] == sys.argv[2] and x['subscription'] == sys.argv[3], x
assert str(x['state']).lower() == 'enabled', x
PY
    then return 0; fi
    sleep 60
  done
}

wait_quota() {
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$state_dir/quota_before_submission.json" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1])); q = row['quotas_(used/limit)']
available = int(q['standard']['limit']) - int(q['standard']['used'])
user_available = int(q['user_max']['limit']) - int(q['user_max'].get('used') or 0)
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps({
    'selected_row': row, 'required_units': 2,
    'standard_available': available, 'user_available': user_available,
}, indent=2, sort_keys=True) + '\n')
assert available >= 2 and user_available >= 2
PY
    then return 0; fi
    sleep 120
  done
}

wait_pass() {
  local experiment="$1"
  while true; do
    heartbeat
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
    if grep -Fqi 'No jobs found in the current experiment' <<< "$status"; then sleep 30; continue; fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 2) else 1)
PY
    then return 0; fi
    return 1
  done
}

deadline=$(( $(date +%s) + 2592000 ))
for branch in "${branches[@]}"; do
  remote_root="studentmodeling/diagnostics/world_model_v1_m4_causal_${branch}_seed42_round500_20260731"
  while [[ "$(az storage blob exists --account-name "$account" --container-name "$container" \
      --auth-mode login --name "$remote_root/checkpoint-round-250/checkpoint_completion.json" \
      --query exists -o tsv 2>/dev/null)" != true ]]; do
    heartbeat
    (( $(date +%s) < deadline )) || fail "checkpoint wait exceeded 30 days: $branch"
    sleep 60
  done
done
log "both directional-tail round-250 atomic checkpoints detected"

configs=() jobs=()
for branch in "${branches[@]}"; do
  suffix="${branch#bootstrap_balanced_sum_}"
  job="m4-directional-${suffix}-checkpoint250-c4-seed42"
  config="$project_dir/amlt/feeds_m4_directional_${suffix}_checkpoint250_c4_20260731.yaml"
  run_dir="/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m4_causal_${branch}_seed42_round500_20260731"
  python scripts/render_checkpoint_c4_probe.py --base "$base_config" --round 250 \
    --run-dir "$run_dir" --job-name "$job" --output "$config" \
    >> "$state_dir/controller.log" 2>&1 || fail "render failed: $branch"
  python scripts/validate_checkpoint_c4_probe_amlt_config.py --config "$config" --round 250 \
    --run-dir "$run_dir" --job-name "$job" \
    >> "$state_dir/controller.log" 2>&1 || fail "validation failed: $branch"
  configs+=("$config")
  jobs+=("$job")
done

# Merge two individually validated template-derived jobs into one parallel AMLT experiment.
combined="$project_dir/amlt/feeds_m4_directional_checkpoint250_c4_20260731.yaml"
python - "$combined" "${configs[@]}" <<'PY'
import copy, sys, yaml
output, *paths = sys.argv[1:]
values = [yaml.safe_load(open(path, encoding='utf-8')) for path in paths]
base = copy.deepcopy(values[0]); base['jobs'] = [value['jobs'][0] for value in values]
base['description'] = 'Parallel frozen 150-pair C4 probes for directional-tail round-250 checkpoints'
open(output, 'w', encoding='utf-8').write(yaml.safe_dump(base, sort_keys=False))
PY

wait_auth
for recovery in 0 1 2 3; do
  experiment="recurrent-m4-directional-checkpoint250-c4-seed42-20260731"
  (( recovery == 0 )) || experiment="${experiment}-recovery${recovery}"
  class="$(experiment_class "$experiment")" || fail "ambiguous experiment: $experiment"
  if [[ "$class" == missing ]]; then
    wait_quota
    exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
    amlt run "$combined" --dump -t Feeds > "$state_dir/${experiment}.dump.yaml" 2>&1 \
      || { flock -u 8; fail "AMLT dump failed"; }
    # Validate each job after AMLT normalization by slicing the dump.
    python - "$state_dir/${experiment}.dump.yaml" "$state_dir" "${jobs[@]}" <<'PY'
import pathlib, sys, yaml
source, root, *jobs = sys.argv[1:]
text = open(source, encoding='utf-8').read(); start = min(i for m in ('code:', 'description:', 'target:') if (i := text.find(m)) >= 0)
value = yaml.safe_load(text[start:])
for job in jobs:
    selected = dict(value); selected['jobs'] = [x for x in value['jobs'] if x['name'] == job]
    pathlib.Path(root, job + '.dump.yaml').write_text(yaml.safe_dump(selected, sort_keys=False))
PY
    for index in 0 1; do
      branch="${branches[$index]}"; suffix="${branch#bootstrap_balanced_sum_}"
      run_dir="/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m4_causal_${branch}_seed42_round500_20260731"
      python scripts/validate_checkpoint_c4_probe_amlt_config.py \
        --config "$state_dir/${jobs[$index]}.dump.yaml" --round 250 \
        --run-dir "$run_dir" --job-name "${jobs[$index]}" \
        >> "$state_dir/controller.log" 2>&1 || { flock -u 8; fail "dump validation failed"; }
    done
    amlt run "$combined" "$experiment" -t Feeds -y \
      --description "Frozen directional-tail round-250 C4; recovery=$recovery" \
      > "$state_dir/${experiment}.submission.txt" 2>&1 \
      || { flock -u 8; fail "submission failed"; }
    flock -u 8
  elif [[ "$class" != exists ]]; then
    fail "ambiguous pre-existing experiment: $experiment"
  fi
  if wait_pass "$experiment"; then
    passed=true
    break
  fi
done
[[ "${passed:-false}" == true ]] || fail "directional round-250 probes exhausted recovery waves"

download="$state_dir/download/$experiment"
mkdir -p "$download"
amlt results download "$experiment" --status pass --include recurrent_causal_gate.json \
  --include probe_scope.txt --output "$download" >> "$state_dir/controller.log" 2>&1 \
  || fail "result download failed"
python - "$download" "$state_dir/summary.json" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); rows = []
for path in sorted(root.rglob('recurrent_causal_gate.json')):
    value = json.load(open(path, encoding='utf-8')); summary = value['summary']; clean = summary['clean']
    assert value['raw_history_visible_to_decoder'] is False
    assert value['persistent_history_kv_cache'] is False
    rows.append({
        'job': path.parents[2].name, 'D_specific': summary['D_specific'],
        'D_damage': summary['D_damage'], 'clean_n': summary['clean_n'],
        'swap': clean['swap']['flip_to_A_rate'], 'random': clean['random']['flip_to_A_rate'],
        'noop': clean['noop']['flip_to_A_rate'], 'same_rule': clean['samemal']['flip_to_A_rate'],
        'scientific_gate_pass': summary['scientific_gate_pass'],
    })
assert len(rows) == 2, rows
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps({'round': 250, 'branches': rows}, indent=2, sort_keys=True) + '\n')
print(json.dumps(rows, indent=2, sort_keys=True))
PY
touch "$state_dir/complete"
log "directional-tail round-250 frozen C4 probes complete"
