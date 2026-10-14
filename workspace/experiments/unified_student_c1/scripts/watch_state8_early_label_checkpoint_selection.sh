#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_early_label_checkpoint_selection_20260813"
screens="$project/results/state8_curriculum_repair_screens_20260812"
full="$project/results/state8_early_label_binding_repairs_20260813"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

for step in 20 100 240 500; do
  inputs=()
  for spec in 8301:early_label_low 8302:early_label_high; do
    seed="${spec%%:*}"; branch="${spec#*:}"
    directory="$screens/seed${seed}_${branch}/checkpoint-$step"
    session="state8_el_${seed}_cp${step}_screen"
    inputs+=("$directory/summary.json")
    if [[ ! -f "$directory/complete" ]] && ! tmux has-session -t "$session" 2>/dev/null; then
      mkdir -p "$directory"
      tmux new-session -d -s "$session" \
        "cd '$project' && REPAIR_SEED=$seed REPAIR_BRANCH=$branch CHECKPOINT_STEP=$step exec bash scripts/watch_state8_curriculum_repair_checkpoint_screen.sh 2>&1 | tee -a '$directory/controller.tmux.log'"
      log "launched early-label adaptive screen seed=$seed branch=$branch step=$step"
    fi
  done
  while :; do
    heartbeat; ready=1
    for path in "${inputs[@]}"; do
      directory="$(dirname "$path")"
      [[ -s "$path" && -f "$directory/complete" ]] || ready=0
      [[ ! -f "$directory/needs_attention" ]] || fail "screen needs attention: $directory"
    done
    (( ready )) && break
    sleep_unlocked 30
  done
  set +e
  "$python_bin" scripts/select_state8_early_label_checkpoint.py \
    --candidate "${inputs[0]}" --candidate "${inputs[1]}" \
    --output "$state/checkpoint-${step}.selection.json" \
    > "$state/checkpoint-${step}.selection.log" 2>&1
  rc=$?
  set -e
  if (( rc == 0 )); then
    cp "$state/checkpoint-${step}.selection.json" "$state/selection.json"
    touch "$state/scientific_gate_pass" "$state/complete"; heartbeat
    log "early-label adaptive selection passed at checkpoint-$step"
    exit 0
  elif (( rc != 2 )); then
    fail "early-label selector crashed at checkpoint-$step rc=$rc"
  fi
  log "early-label checkpoint-$step pair failed shared gates; advancing"
done

# The four bounded milestones can all fail while the full 1000-step tail later
# recovers.  Reuse the full semantic selection rather than inventing a weaker
# terminal rule.
while [[ ! -f "$full/complete" ]]; do
  heartbeat
  [[ ! -f "$full/needs_attention" ]] || fail "full early-label tail needs attention"
  sleep_unlocked 30
done
[[ -s "$full/selection.json" ]] || fail "full early-label selection missing"
cp "$full/selection.json" "$state/selection.json"
if "$python_bin" - "$state/selection.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('pass') is True else 2)
PY
then
  touch "$state/scientific_gate_pass"
  log "full early-label tail passed after all adaptive milestones failed"
else
  touch "$state/scientific_gate_failed"
  log "all early-label milestones and full tail failed shared gates"
fi
touch "$state/complete"; heartbeat
