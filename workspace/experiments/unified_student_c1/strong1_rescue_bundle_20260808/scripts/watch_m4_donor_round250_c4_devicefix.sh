#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/donor_round250_c4_devicefix"
account="csnewsandfeeds4150361735"
container="amulet"
branches=(donor_directional_sum_w1 donor_directional_sum_w4)
experiments=(
  recurrent-m4-donor-w1-round250-c4-devicefix-seed42-20260731
  recurrent-m4-donor-w4-round250-c4-devicefix-seed42-20260731
)
configs=(
  "$project_dir/amlt/feeds_m4_donor_w1_round250_c4_devicefix_20260731.yaml"
  "$project_dir/amlt/feeds_m4_donor_w4_round250_c4_devicefix_20260731.yaml"
)
mkdir -p "$state_dir/download"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail() { touch "$state_dir/needs_attention"; exit 1; }

submitted=(false false)
passed=(false false)
while true; do
  heartbeat
  all_submitted=true
  for index in 0 1; do
    [[ "${submitted[$index]}" != true ]] || continue
    branch="${branches[$index]}"
    remote="studentmodeling/diagnostics/world_model_v1_m4_causal_${branch}_seed42_round500_20260731/checkpoint-round-250/checkpoint_completion.json"
    exists="$(az storage blob exists --account-name "$account" --container-name "$container" \
      --auth-mode login --name "$remote" --query exists -o tsv 2>/dev/null || true)"
    if [[ "$exists" != true ]]; then
      all_submitted=false
      continue
    fi
    experiment="${experiments[$index]}"
    config="${configs[$index]}"
    status="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
    class="$(python - "$rc" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || fail
    if [[ "$class" == missing ]]; then
      exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
      amlt run "$config" "$experiment" -t Feeds -y \
        --description "Mid-curriculum C4-only probe of $branch round-250; no retraining" \
        > "$state_dir/${experiment}.submission.txt" 2>&1 \
        || { flock -u 8; fail; }
      flock -u 8
    elif [[ "$class" != exists ]]; then
      fail
    fi
    submitted[$index]=true
  done
  [[ "$all_submitted" == true ]] && break
  sleep 60
done

while true; do
  heartbeat
  all_passed=true
  for index in 0 1; do
    [[ "${passed[$index]}" != true ]] || continue
    experiment="${experiments[$index]}"
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      all_passed=false
      continue
    fi
    python - "$status" <<'PY' || fail
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    amlt results download "$experiment" --status pass --include recurrent_causal_gate.json \
      --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 || fail
    passed[$index]=true
  done
  [[ "$all_passed" == true ]] && break
  sleep 120
done

python - "$state_dir/download" "$state_dir/summary.json" <<'PY'
import json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
rows=[]
for path in sorted(root.rglob('recurrent_causal_gate.json')):
    x=json.loads(path.read_text()); s=x['summary']; c=s['clean']
    job=next(part for part in path.parts if part.startswith('m4-donor-'))
    rows.append({'job':job,'clean_n':s['clean_n'],'D_specific':s['D_specific'],
                 'D_damage':s['D_damage'],'swap':c['swap']['flip_to_A_rate'],
                 'random':c['random']['flip_to_A_rate'],
                 'noop':c['noop']['flip_to_A_rate'],
                 'same_rule':c['samemal']['flip_to_A_rate'],
                 'scientific_gate_pass':s['scientific_gate_pass']})
if len(rows)!=2: raise SystemExit(f'expected 2 results, got {len(rows)}')
output.write_text(json.dumps({'round':250,'branches':rows},indent=2,sort_keys=True)+'\n')
PY
touch "$state_dir/complete"
