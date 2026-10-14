#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/cf4_diagnostic"
config="$project_dir/amlt/feeds_m4_cf4_diagnostic_20260731.yaml"
base_experiment="recurrent-m4-cf4-diagnostic-seed42-20260731"
experiment="$base_experiment"
expected_jobs=(
  m4-causal-cf4-bootstrap-symmetric-w1-round0to250-seed42
  m4-causal-cf4-donor-directional-w1-round0to250-seed42
)
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

active_jobs=("${expected_jobs[@]}")

download_passed() {
  local source_experiment="$1" source_status="$2"
  grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$source_status" || return 0
  amlt results download "$source_experiment" --status pass \
    --include recurrent_causal_gate.json \
    --include validation_metrics.json \
    --include causal_validation_metrics.json \
    --include branch_training_summary.json \
    --include run_config.json \
    --include completion.json \
    --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1
}

passed=false
for recovery_wave in 0 1 2 3; do
  while true; do
    heartbeat
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out' <<< "$status"; then
      sleep 60
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$status" "${#active_jobs[@]}" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
    then
      download_passed "$experiment" "$status" || fail "passed-result download failed"
      passed=true
      break
    fi
    break
  done
  [[ "$passed" != true ]] || break
  printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
  download_passed "$experiment" "$status" || fail "partial passed-result download failed"
  timeout 180 amlt logs view -n 500 "$experiment" \
    > "$state_dir/${experiment}.failure.log" 2>&1 || true
  mapfile -t failed_jobs < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]), sep='\n')
PY
  )
  (( ${#failed_jobs[@]} > 0 )) || fail "terminal CF4 state has no recoverable jobs"
  (( recovery_wave < 3 )) || break
  experiment="${base_experiment}-recovery$((recovery_wave + 1))"
  active_jobs=("${failed_jobs[@]}")
  selectors=()
  for job in "${active_jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$state_dir/${experiment}.dump.yaml" 2>&1 \
    || { flock -u 8; fail "recovery dump failed"; }
  python scripts/validate_m4_cf4_diagnostic_amlt_config.py \
    --config "$state_dir/${experiment}.dump.yaml" --jobs "${active_jobs[@]}" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; fail "recovery preflight failed"; }
  output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "CF4 recurrent causal diagnostic targeted recovery $((recovery_wave + 1)); jobs=${active_jobs[*]}; scientific protocol unchanged" 2>&1)"
  rc=$?
  printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || fail "recovery submission failed"
done
[[ "$passed" == true ]] || fail "CF4 diagnostic exhausted infrastructure recovery"

printf '%s\n' "$status" > "$state_dir/terminal_status.txt"

python - "$state_dir/download" "$state_dir/comparison.json" "$state_dir/comparison.md" <<'PY'
import json, pathlib, sys
root, output, markdown = map(pathlib.Path, sys.argv[1:])
paths = sorted(root.rglob('recurrent_causal_gate.json'))
if len(paths) != 2:
    raise SystemExit(f'expected two C4 results, found {len(paths)}')
jobs = [next(part for part in path.parts if part.startswith('m4-causal-cf4-')) for path in paths]
if len(set(jobs)) != 2:
    raise SystemExit(f'expected two distinct C4 jobs, found {jobs}')
rows=[]
for path in paths:
    job=next(part for part in path.parts if part.startswith('m4-causal-cf4-'))
    job_root=path.parent.parent.parent
    matches=list(job_root.rglob('validation_metrics.json'))
    causal_matches=list(job_root.rglob('causal_validation_metrics.json'))
    summary_matches=list(job_root.rglob('branch_training_summary.json'))
    if len(matches)!=1 or len(causal_matches)!=1 or len(summary_matches)!=1:
        raise SystemExit(f'incomplete metrics for {job}')
    gate=json.loads(path.read_text())['summary']
    validation=json.loads(matches[0].read_text())
    causal=json.loads(causal_matches[0].read_text())
    training=json.loads(summary_matches[0].read_text())
    clean=gate['clean']
    rows.append({
      'job':job,
      'branch':training['branch'],
      'C4_D_specific':float(gate['D_specific']),
      'C4_D_damage':float(gate['D_damage']),
      'C4_clean_n':int(gate['clean_n']),
      'C4_swap_flip':float(clean['swap']['flip_to_A_rate']),
      'C4_random_flip':float(clean['random']['flip_to_A_rate']),
      'C4_noop_flip':float(clean['noop']['flip_to_A_rate']),
      'C4_same_rule_flip':float(clean['samemal']['flip_to_A_rate']),
      'validation_accuracy_gap':float(validation['accuracy_gap_recurrent_minus_direct']),
      'validation_recurrent_accuracy':float(validation['recurrent_accuracy']),
      'causal_valid_directed_flip':float(causal['directed_flip_to_A_on_clean']),
      'causal_valid_native_accuracy':float(causal['native_accuracy']),
      'causal_valid_donor_accuracy':float(causal['donor_accuracy']),
      'eligible': bool(
        gate['clean_n'] >= 50
        and clean['noop']['flip_to_A_rate'] <= .05
        and clean['samemal']['flip_to_A_rate'] <= .10
        and validation['accuracy_gap_recurrent_minus_direct'] >= -.05
      ),
    })
eligible=[r for r in rows if r['eligible']]
selected=max(eligible,key=lambda r:r['C4_D_specific'],default=None)
payload={
  'stage':'M4_CF4_multiquery_diagnostic',
  'branches':rows,
  'selected_branch':selected['branch'] if selected and selected['C4_D_specific']>=.15 else None,
  'scientific_gate_pass':bool(selected and selected['C4_D_specific']>=.15),
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
lines=['# CF4 multi-query recurrent causal diagnostic','',
 '| branch | D_specific | swap | random | clean n | validation gap | causal-valid flip |',
 '|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 lines.append(f"| {r['branch']} | {r['C4_D_specific']:.3f} | {r['C4_swap_flip']:.3f} | {r['C4_random_flip']:.3f} | {r['C4_clean_n']} | {r['validation_accuracy_gap']:.3f} | {r['causal_valid_directed_flip']:.3f} |")
markdown.write_text('\n'.join(lines)+'\n')
PY

touch "$state_dir/complete"
log "CF4 comparison complete"
