#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_curriculum_selection_20260812"
screens="$project/results/state8_curriculum_screens_20260812"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

inputs=()
for arm in generic current_query; do
  for step in 2000 3000 4000 5000; do
    inputs+=("$screens/$arm/checkpoint-$step/summary.json")
  done
done
while :; do
  heartbeat; ready=1
  for path in "${inputs[@]}"; do
    parent="$(dirname "$path")"
    if [[ -s "$path" ]]; then
      # A completed checkpoint screen is valid selector input even when an
      # earlier infrastructure attempt left a stale needs_attention marker.
      # Scientific gate failures belong in the selector's failed_checks; they
      # are not orchestration failures.
      continue
    fi
    ready=0
    [[ ! -f "$parent/needs_attention" ]] \
      || fail "checkpoint screen needs attention before producing summary path=$parent"
  done
  (( ready )) && break
  sleep_unlocked 60
done

args=(); for path in "${inputs[@]}"; do args+=(--candidate "$path"); done
set +e
"$python_bin" scripts/select_state8_curriculum_checkpoint.py "${args[@]}" \
  --output "$state/selection.json" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  touch "$state/scientific_gate_pass"
  log "curriculum checkpoint joint selection passed"
elif (( rc == 2 )); then
  touch "$state/scientific_gate_failed"
  log "all curriculum checkpoints failed at least one hard gate"
else
  fail "curriculum checkpoint selector crashed rc=$rc"
fi
touch "$state/complete"; heartbeat
