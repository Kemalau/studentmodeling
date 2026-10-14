#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_causal_teacher_checkpoint_selection_20260812"
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
  8201:causal_teacher_low:20
  8201:causal_teacher_low:100
  8201:causal_teacher_low:240
  8201:causal_teacher_low:500
  8202:causal_teacher_high:20
  8202:causal_teacher_high:100
  8202:causal_teacher_high:240
  8202:causal_teacher_high:500
)
inputs=()
for spec in "${specs[@]}"; do
  IFS=: read -r seed branch step <<< "$spec"
  directory="$screens/seed${seed}_${branch}/checkpoint-$step"
  case "$seed:$step" in
    8201:20) session=state8_ct_low_cp20_screen ;;
    8201:100) session=state8_ct_low_cp100_screen ;;
    8201:240) session=state8_ct_low_cp240_screen ;;
    8201:500) session=state8_ct_low_cp500_screen ;;
    8202:20) session=state8_ct_high_cp20_screen ;;
    8202:100) session=state8_ct_high_cp100_screen ;;
    8202:240) session=state8_ct_high_cp240_screen ;;
    8202:500) session=state8_ct_high_cp500_screen ;;
    *) fail "unsupported causal-teacher screen coordinate seed=$seed step=$step" ;;
  esac
  inputs+=("$directory/summary.json")
  if [[ ! -f "$directory/complete" ]] && ! tmux has-session -t "$session" 2>/dev/null; then
    mkdir -p "$directory"
    tmux new-session -d -s "$session" \
      "cd '$project' && REPAIR_SEED=$seed REPAIR_BRANCH=$branch CHECKPOINT_STEP=$step exec bash scripts/watch_state8_curriculum_repair_checkpoint_screen.sh"
    log "launched causal-teacher checkpoint screen seed=$seed branch=$branch step=$step"
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
"$python_bin" scripts/select_state8_causal_teacher_checkpoint.py \
  "${args[@]}" --output "$state/selection.json" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  touch "$state/scientific_gate_pass"
  log "causal-teacher checkpoint joint selection passed"
elif (( rc == 2 )); then
  touch "$state/scientific_gate_failed"
  log "all causal-teacher checkpoints failed the shared gates"
else
  fail "causal-teacher checkpoint selector crashed rc=$rc"
fi
touch "$state/complete"; heartbeat
