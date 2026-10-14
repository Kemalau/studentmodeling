#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
state="$project/results/state8_eedi_label_readout_20260813"
config="$project/amlt/feeds_state8_eedi_label_readout_20260813.yaml"
base=state8-eedi-label-readout-20260813
experiment="$base"; recovery=0; max_recovery=2
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
all_jobs=(
  state8-label-readout-early-label-low-cp20
  state8-label-readout-early-label-high-cp20
  state8-label-readout-causal-teacher-low-cp100-control
)

mkdir -p "$state/passed"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

if [[ ! -s "$state/active_experiment.txt" ]]; then
  printf '%s\n' "$base" > "$state/active_experiment.txt"
  printf '%s\n' "${all_jobs[@]}" > "$state/active_jobs.txt"
fi
experiment="$(<"$state/active_experiment.txt")"
[[ "$experiment" =~ -recovery([0-9]+)$ ]] && recovery="${BASH_REMATCH[1]}"

submit_failed_wave(){
  local name="$1"; shift
  local jobs=("$@") audit quota available auditable output rc status
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  audit="$state/$name.active_a100.json"
  "$python_bin" scripts/count_active_a100_requests.py --user-limit 32 --recent 500 \
    --output "$audit" >/dev/null 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  read -r available auditable < <("$python_bin" - "$audit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(int(x['available_user_slots']),int(bool(x['auditable'])))
PY
  )
  [[ "$auditable" == 1 && "$available" -ge "${#jobs[@]}" ]] \
    || { flock -u 8; return 2; }
  quota="$state/$name.quota.json"
  amlt --json-tables ti sing -t Feeds > "$quota" 2>> "$state/controller.log" \
    || { flock -u 8; return 1; }
  "$python_bin" - "$quota" "${#jobs[@]}" <<'PY' || { flock -u 8; return 2; }
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
r=select_a100_80gb(parse_json_table(open(sys.argv[1]).read())); need=int(sys.argv[2])
q=r['quotas_(used/limit)']; assert q['standard']['limit']-q['standard']['used']>=need,q
assert int(q['user_max']['limit'])>=need,q
PY
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_eedi_label_readout_amlt.py \
    --config "$state/$name.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  status="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  if (( rc == 0 )); then
    flock -u 8; return 0
  fi
  selectors=(); for job in "${jobs[@]}"; do selectors+=(":$job"); done
  output="$(amlt run "$config" "${selectors[@]}" "$name" -t Feeds -y --description \
    "Targeted recovery of failed read-only State8 label-readout jobs" 2>&1)"; rc=$?
  printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
    > "$state/$name.submission.txt"
  if (( rc != 0 )); then
    status="$(amlt status --no-update-db "$name" 2>&1)"
    amlt status --no-update-db "$name" >/dev/null 2>&1 \
      || { flock -u 8; return 1; }
  fi
  flock -u 8
  return 0
}

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  mapfile -t active_jobs < "$state/active_jobs.txt"
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls \
    --force-status-update 2>/dev/null)" || { sleep_unlocked 30; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  mapfile -t parsed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
    name=str(row.get('job_name') or '').lstrip(':')
    status=str(row.get('status') or '').casefold()
    print(f'{name}\t{status}')
PY
  )
  active=0; paused=(); failed=(); passed=()
  for job in "${active_jobs[@]}"; do
    phase=""
    for line in "${parsed[@]}"; do
      [[ "${line%%$'\t'*}" == "$job" ]] && phase="${line#*$'\t'}"
    done
    case "$phase" in
      pass|passed) passed+=("$job") ;;
      preparing|pre-queued|queued|running|resuming) active=1 ;;
      paused) active=1; paused+=("$job") ;;
      '') active=1 ;;
      *) failed+=("$job") ;;
    esac
  done
  for job in "${paused[@]}"; do
    timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
  done
  (( active == 1 )) && { sleep_unlocked 30; continue; }

  for job in "${passed[@]}"; do printf '%s\n' "$experiment" > "$state/passed/$job.experiment"; done
  if (( ${#failed[@]} > 0 )); then
    for job in "${failed[@]}"; do
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.$job.failure.log" 2>&1 || true
    done
    (( recovery < max_recovery )) || fail "label-readout exhausted targeted recovery waves"
    recovery=$((recovery+1)); experiment="${base}-recovery${recovery}"
    until submit_failed_wave "$experiment" "${failed[@]}"; do heartbeat; sleep_unlocked 60; done
    printf '%s\n' "$experiment" > "$state/active_experiment.txt"
    printf '%s\n' "${failed[@]}" > "$state/active_jobs.txt"
    log "submitted targeted recovery wave=$recovery jobs=${failed[*]}"
    continue
  fi
  complete=1
  for job in "${all_jobs[@]}"; do [[ -s "$state/passed/$job.experiment" ]] || complete=0; done
  (( complete == 1 )) || fail "terminal wave left missing readout jobs"
  touch "$state/remote_passed"
done

rm -rf "$state/download"; mkdir -p "$state/download"
inputs=()
for job in "${all_jobs[@]}"; do
  source_experiment="$(<"$state/passed/$job.experiment")"
  amlt results download "$source_experiment" ":$job" --include '*.json' \
    --include '*.txt' --output "$state/download" >> "$state/controller.log" 2>&1 \
    || fail "result download failed job=$job experiment=$source_experiment"
  mapfile -t metrics < <(find "$state/download" -type f \
    -path "*/$job/eval/eedi_label_readout_25/metrics.json")
  [[ ${#metrics[@]} -eq 1 ]] || fail "metrics cardinality drift job=$job"
  inputs+=(--input "${metrics[0]}")
done
"$python_bin" scripts/summarize_state8_eedi_label_readout.py "${inputs[@]}" \
  --output "$state/summary.json" >> "$state/controller.log" 2>&1 \
  || fail "label-readout comparison failed"
touch "$state/complete"; heartbeat
log "State8 label-readout comparison complete"
