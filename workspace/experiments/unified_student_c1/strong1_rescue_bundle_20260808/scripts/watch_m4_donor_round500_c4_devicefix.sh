#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/donor_round500_c4_devicefix"
target_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_donor_directional"
training_experiment="recurrent-m4-causal-donor-directional-preemptible-seed42-20260731"
eval_experiment="recurrent-m4-donor-round500-c4-devicefix-seed42-20260731"
config="$project_dir/amlt/feeds_m4_donor_round500_c4_devicefix_20260731.yaml"
training_config="$project_dir/amlt/feeds_m4_causal_donor_directional_preemptible_20260731.yaml"
account="csnewsandfeeds4150361735"
container="amulet"
remote_prefix="studentmodeling/diagnostics/world_model_v1_m4_causal"
branches=(donor_directional_sum_w1 donor_directional_sum_w4)
training_jobs=(
  m4-causal-donor-directional-sum-w1-round0to500-seed42
  m4-causal-donor-directional-sum-w4-round0to500-seed42
)
mkdir -p "$state_dir/download" "$state_dir/training"
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
fail() {
  log "ERROR: $*"
  touch "$state_dir/needs_attention" "$target_dir/needs_attention"
  exit 1
}

# Training jobs publish atomic round-500 checkpoints before the known legacy
# tail evaluator runs.  Wait for both durable checkpoints instead of treating
# the later device-placement failure as a failed optimizer run.  If a job dies
# before publishing its checkpoint, recover only the missing branch.
deadline=$(( $(date +%s) + 604800 ))
recovery_wave=0
# Resume the latest actually submitted recovery experiment after a controller
# restart.  A failed preflight can leave a dump for the next wave without a
# submission receipt, so dumps alone are deliberately ignored.  This prevents
# appending a duplicate branch to an already queued/running recovery experiment.
for candidate_wave in 1 2 3; do
  candidate_experiment="recurrent-m4-causal-donor-directional-devicefix-recovery${candidate_wave}-seed42-20260731"
  if [[ -f "$state_dir/${candidate_experiment}.submission.txt" ]]; then
    recovery_wave="$candidate_wave"
    training_experiment="$candidate_experiment"
  fi
done
while true; do
  heartbeat
  missing_indices=()
  for index in 0 1; do
    branch="${branches[$index]}"
    remote_root="${remote_prefix}_${branch}_seed42_round500_20260731"
    exists="$(az storage blob exists --account-name "$account" \
      --container-name "$container" --auth-mode login \
      --name "$remote_root/checkpoint-round-500/checkpoint_completion.json" \
      --query exists -o tsv 2>/dev/null || true)"
    [[ "$exists" == true ]] || missing_indices+=("$index")
  done
  (( ${#missing_indices[@]} > 0 )) || break
  (( $(date +%s) < deadline )) || fail "checkpoint wait exceeded"
  status="$(amlt status "$training_experiment" 2>&1)"
  printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$training_experiment" "$status" >> "$state_dir/training_status.log"
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60
    continue
  fi
  (( recovery_wave < 3 )) || fail "training checkpoint recovery exhausted"
  recovery_wave=$((recovery_wave + 1))
  recovery_jobs=()
  selectors=()
  for index in "${missing_indices[@]}"; do
    recovery_jobs+=("${training_jobs[$index]}")
    selectors+=(":${training_jobs[$index]}")
  done
  training_experiment="recurrent-m4-causal-donor-directional-devicefix-recovery${recovery_wave}-seed42-20260731"
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  amlt run "$training_config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${training_experiment}.dump.yaml" 2>&1 \
    || { flock -u 8; fail "training recovery dump failed"; }
  python scripts/validate_m4_donor_preemptible_amlt_config.py \
    --config "$state_dir/${training_experiment}.dump.yaml" \
    --jobs "${recovery_jobs[@]}" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "training recovery preflight failed"; }
  output="$(amlt run "$training_config" "${selectors[@]}" "$training_experiment" \
    -t Feeds -y --description "Checkpoint-targeted donor-directional recovery; jobs=${recovery_jobs[*]}" 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state_dir/${training_experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || fail "training recovery submission failed"
done
log "both donor-directional round-500 checkpoints detected"

# The legacy in-job C4 evaluator may fail after training because it sees all
# four GPUs and shards the backbone.  Preserve every durable training artifact
# first, then stop only the now-redundant tail process to release eight GPUs for
# the fixed one-GPU evaluations.
for branch in "${branches[@]}"; do
  remote_root="${remote_prefix}_${branch}_seed42_round500_20260731"
  for name in completion.json branch_training_summary.json validation_metrics.json \
      causal_validation_metrics.json final_adapter/adapter_model.safetensors \
      recurrent_state.pt; do
    while [[ "$(az storage blob exists --account-name "$account" \
        --container-name "$container" --auth-mode login --name "$remote_root/$name" \
        --query exists -o tsv 2>/dev/null)" != true ]]; do
      heartbeat
      sleep 30
    done
  done
done
for stale_experiment in \
    recurrent-m4-causal-donor-directional-seed42-20260731 \
    recurrent-m4-causal-donor-directional-preemptible-seed42-20260731 \
    recurrent-m4-causal-donor-directional-devicefix-recovery1-seed42-20260731 \
    recurrent-m4-causal-donor-directional-devicefix-recovery2-seed42-20260731 \
    recurrent-m4-causal-donor-directional-devicefix-recovery3-seed42-20260731; do
  amlt cancel "$stale_experiment" --status running -y \
    >> "$state_dir/controller.log" 2>&1 || true
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/durable_training_artifacts_preserved"
log "durable donor training artifacts preserved; redundant legacy tail stopped"

class="$(python - <<'PY'
import subprocess
from scripts.recurrent_submit_guard import classify_experiment_status
p = subprocess.run(
    ['amlt', 'status', '--no-update-db', 'recurrent-m4-donor-round500-c4-devicefix-seed42-20260731'],
    text=True, capture_output=True,
)
print(classify_experiment_status(p.returncode, p.stdout + p.stderr))
PY
)" || fail "could not classify C4 experiment"
if [[ "$class" == missing ]]; then
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  output="$(amlt run "$config" "$eval_experiment" -t Feeds -y \
    --description 'C4-only evaluation of completed donor-directional checkpoints after explicit sharded-device state fix; no retraining' 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state_dir/submission.txt"
  flock -u 8
  (( rc == 0 )) || fail "C4 submission failed"
elif [[ "$class" != exists ]]; then
  fail "ambiguous C4 experiment state"
fi

while true; do
  heartbeat
  status="$(amlt status "$eval_experiment" 2>&1)"
  printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$eval_experiment" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out' <<< "$status"; then
    sleep 60
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
  timeout 180 amlt logs view -n 300 "$eval_experiment" \
    > "$state_dir/failure.log" 2>&1 || true
  fail "C4 evaluation reached a non-pass terminal state"
done

amlt results download "$eval_experiment" --status pass \
  --include recurrent_causal_gate.json --include probe_scope.txt \
  --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1 \
  || fail "result download failed"

for branch in "${branches[@]}"; do
  branch_dir="$state_dir/training/$branch"
  remote_root="${remote_prefix}_${branch}_seed42_round500_20260731"
  mkdir -p "$branch_dir"
  for name in branch_training_summary.json validation_metrics.json \
      causal_validation_metrics.json; do
    az storage blob download --account-name "$account" \
      --container-name "$container" --auth-mode login \
      --name "$remote_root/$name" --file "$branch_dir/$name" --overwrite \
      >> "$state_dir/controller.log" 2>&1 \
      || fail "could not download $branch/$name"
  done
done

python - "$state_dir/download" "$state_dir/training" \
  "$state_dir/comparison.json" "$state_dir/comparison.md" <<'PY'
import json
import pathlib
import sys

root, training_root, output, output_md = map(pathlib.Path, sys.argv[1:])
paths = sorted(root.rglob('recurrent_causal_gate.json'))
if len(paths) != 2:
    raise SystemExit(f'expected two C4 results, found {len(paths)}')
branches = []
for path in paths:
    result = json.loads(path.read_text(encoding='utf-8'))
    summary = result['summary']
    job = next(part for part in path.parts if part.startswith('m4-donor-'))
    branch_name = 'donor_directional_sum_w1' if '-w1-' in job else 'donor_directional_sum_w4'
    train_dir = training_root / branch_name
    branch = json.loads((train_dir / 'branch_training_summary.json').read_text())
    validation = json.loads((train_dir / 'validation_metrics.json').read_text())
    causal = json.loads((train_dir / 'causal_validation_metrics.json').read_text())
    clean = summary['clean']
    branches.append({
        **branch,
        'C4_D_specific': float(summary['D_specific']),
        'C4_D_damage': float(summary['D_damage']),
        'C4_clean_n': int(summary['clean_n']),
        'C4_swap_flip': float(clean['swap']['flip_to_A_rate']),
        'C4_random_flip': float(clean['random']['flip_to_A_rate']),
        'C4_noop_flip': float(clean['noop']['flip_to_A_rate']),
        'C4_same_rule_flip': float(clean['samemal']['flip_to_A_rate']),
        'validation_recurrent_accuracy': float(validation['recurrent_accuracy']),
        'validation_direct_accuracy': float(validation['direct_accuracy']),
        'validation_accuracy_gap': float(validation['accuracy_gap_recurrent_minus_direct']),
        'causal_native_accuracy': float(causal['native_accuracy']),
        'causal_donor_accuracy': float(causal['donor_accuracy']),
        'causal_directed_flip': float(causal['directed_flip_to_A_on_clean']),
        'source': str(path),
    })
eligible = [
    row for row in branches
    if row['C4_clean_n'] >= 50
    and row['C4_noop_flip'] <= 0.05
    and row['C4_same_rule_flip'] <= 0.10
    and row['validation_accuracy_gap'] >= -0.05
]
selected = max(eligible, key=lambda row: row['C4_D_specific'], default=None)
payload = {
    'stage': 'M4_causal_strength_round500',
    'branches': branches,
    'selected_branch': selected['branch'] if selected and selected['C4_D_specific'] >= 0.15 else None,
    'scientific_gate_pass': bool(selected and selected['C4_D_specific'] >= 0.15),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
lines = [
    '# Donor-directional round-500 C4 (device-fixed evaluation)', '',
    '| branch | C4 D_specific | C4 damage | swap | random | clean n | recurrent acc | direct acc |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
]
for row in branches:
    lines.append(
        f"| {row['branch']} | {row['C4_D_specific']:.3f} | {row['C4_D_damage']:.3f} | "
        f"{row['C4_swap_flip']:.3f} | {row['C4_random_flip']:.3f} | {row['C4_clean_n']} | "
        f"{row['validation_recurrent_accuracy']:.3f} | {row['validation_direct_accuracy']:.3f} |"
    )
output_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

cp "$state_dir/comparison.json" "$target_dir/comparison.json"
cp "$state_dir/comparison.md" "$target_dir/comparison.md"
rm -f "$target_dir/needs_attention"
touch "$target_dir/complete" "$state_dir/complete"
log "donor-directional C4 comparison complete"
