#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_curriculum_repair_checkpoint_selection_20260812"
screens="$project/results/state8_curriculum_repair_screens_20260812"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

specs=(
  5201:semantic_binding:state8_repair_5201_sem_cp500_screen
  5202:specificity_triplet:state8_repair_5202_spec_cp500_screen
  5203:balanced_continuation:state8_repair_5203_bal_cp500_screen
  6201:semantic_binding:state8_repair_6201_sem_cp500_screen
  6202:specificity_triplet:state8_repair_6202_spec_cp500_screen
  6203:balanced_continuation:state8_repair_6203_bal_cp500_screen
)
inputs=()
for spec in "${specs[@]}"; do
  IFS=: read -r seed branch session <<< "$spec"
  directory="$screens/seed${seed}_${branch}/checkpoint-500"
  inputs+=("$directory/summary.json")
  if [[ ! -f "$directory/complete" ]] && ! tmux has-session -t "$session" 2>/dev/null; then
    mkdir -p "$directory"
    tmux new-session -d -s "$session" \
      "cd '$project' && REPAIR_SEED=$seed REPAIR_BRANCH=$branch CHECKPOINT_STEP=500 exec bash scripts/watch_state8_curriculum_repair_checkpoint_screen.sh"
    log "launched repair checkpoint screen seed=$seed branch=$branch step=500"
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

args=(); for path in "${inputs[@]}"; do args+=(--candidate "$path"); done
set +e
"$python_bin" scripts/select_state8_curriculum_repair_checkpoint.py \
  "${args[@]}" --output "$state/selection.json" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  touch "$state/scientific_gate_pass"
  log "repair checkpoint joint selection passed"
elif (( rc == 2 )); then
  touch "$state/scientific_gate_failed"
  log "all cp3000/cp4000 repair checkpoint candidates failed the shared gates"
else
  fail "repair checkpoint selector crashed rc=$rc"
fi
touch "$state/complete"; heartbeat
