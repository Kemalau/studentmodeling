#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
base_experiment=promoted-round1750-c4-n150-multiseed-20260804
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1750_c4_n150_multiseed"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
config="$project_dir/amlt/feeds_promoted_round1750_c4_n150_multiseed_20260804.yaml"
active_file="$state_dir/active_experiment.txt"
jobs_file="$state_dir/active_jobs.txt"
preserved="$state_dir/preserved_results"
all_jobs=(
  promoted-round1750-c4-n150-seed13
  promoted-round1750-c4-n150-seed42
  promoted-round1750-c4-n150-seed97
)

mkdir -p "$state_dir" "$preserved"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
preserve_passed(){
  local experiment="$1" status
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || return 1
  grep -Eqi '(^|[[:space:]])pass([[:space:]]|$)' <<< "$status" || return 0
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.txt' --output "$preserved" \
    >> "$state_dir/controller.log" 2>&1
}
wait_quota(){
  local required="$1" quota
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if "$python_bin" - "$quota" "$required" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
q=select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']; n=int(sys.argv[2])
assert int(q['standard']['limit'])-int(q['standard']['used'])>=n
assert int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)>=n
PY
    then return 0; fi
    sleep 120
  done
}
submit_recovery(){
  local experiment="$1"; shift
  local names=("$@") selectors=() output current class rc
  for item in "${names[@]}"; do selectors+=(":$item"); done
  wait_quota "${#names[@]}"
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_promoted_round1750_c4_multiseed_amlt.py \
    --config "$state_dir/${experiment}.preflight.dump.yaml" --jobs "${names[@]}" \
    >> "$state_dir/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
      --description "Round-1750 C4 failed-sibling recovery jobs=${names[*]}; frozen checkpoint and protocol unchanged" 2>&1)"; rc=$?
    printf '[%s] submit rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" == exists ]]; then
    printf 'status=preexisting_not_resubmitted\n%s\n' "$current" \
      > "$state_dir/${experiment}.submission.txt"
  else
    flock -u 8; return 1
  fi
  flock -u 8
  printf '%s\n' "$experiment" > "$active_file"
  printf '%s\n' "${names[@]}" > "$jobs_file"
}

rm -f "$state_dir/complete" "$state_dir/needs_attention"
if [[ ! -s "$active_file" ]]; then
  printf '%s\n' "$base_experiment" > "$active_file"
  printf '%s\n' "${all_jobs[@]}" > "$jobs_file"
fi
experiment="$(<"$active_file")"
expected_jobs="$(wc -l < "$jobs_file")"
wave=0
if [[ "$experiment" =~ recovery([0-9]+)-20260804 ]]; then wave="${BASH_REMATCH[1]}"; fi
while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60
    continue
  fi
  if "$python_bin" - "$status" "$expected_jobs" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], int(sys.argv[2])) else 1)
PY
  then
    preserve_passed "$experiment" || fail "could not preserve passed round-1750 results"
    break
  fi
  preserve_passed "$experiment" || fail "could not preserve passed round-1750 siblings"
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
  )
  (( ${#failed[@]} > 0 )) || fail "terminal round-1750 status has no failed jobs"
  for job in "${failed[@]}"; do
    amlt logs view -n 1000 "$experiment" ":$job" \
      > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
  done
  (( wave < 5 )) || fail "round-1750 C4 exhausted five recovery waves"
  wave=$((wave+1))
  experiment="promoted-round1750-c4-n150-recovery${wave}-20260804"
  submit_recovery "$experiment" "${failed[@]}" \
    || fail "round-1750 failed-sibling recovery submission failed wave=$wave"
  expected_jobs="${#failed[@]}"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
cp -a "$preserved/." "$state_dir/download/"

readarray -t summaries < <(find "$state_dir/download" -type f -path '*/eval/c4/summary.json' | sort)
[[ ${#summaries[@]} -eq 3 ]] || fail "downloaded round-1750 multiseed C4 summary cardinality drift"
"$python_bin" - "$state_dir/summary.json" "${summaries[@]}" <<'PY' \
  || fail "downloaded round-1750 multiseed n150 C4 audit failed"
import json,pathlib,statistics,sys
out=pathlib.Path(sys.argv[1]); rows=[json.load(open(path)) for path in sys.argv[2:]]
assert {row['pair_seed'] for row in rows}=={13,42,97},rows
assert all(row['checkpoint_round']==1750 and row['all_pairs_n']==150 for row in rows),rows
rows=sorted(rows,key=lambda row:row['pair_seed'])
values=[row['D_specific'] for row in rows]
summary={
  'stage':'layerwise_checkpoint_c4_multiseed',
  'checkpoint_round':1750,
  'diagnostic_only':True,
  'pair_seeds':[13,42,97],
  'rows':rows,
  'D_specific_mean':statistics.mean(values),
  'D_specific_min':min(values),
  'scientific_gate_pass_count':sum(bool(row['scientific_gate_pass']) for row in rows),
}
summary['two_of_three_gate_pass']=summary['scientific_gate_pass_count']>=2
temporary=out.with_suffix('.json.partial')
temporary.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
temporary.replace(out)
print(json.dumps(summary,indent=2,sort_keys=True))
PY
touch "$state_dir/complete"
heartbeat
log "promoted round-1750 multiseed n150 C4 complete and audited"
